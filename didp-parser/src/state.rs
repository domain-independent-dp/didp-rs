use crate::variable::{Continuous, Element, Integer, Numeric, Permutation, Set};
use crate::yaml_util;
use lazy_static::lazy_static;
use ordered_float::OrderedFloat;
use std::cmp::Ordering;
use std::collections;
use std::fmt;
use std::rc::Rc;
use std::str;

#[derive(Debug, PartialEq, Clone, Default)]
pub struct State<T: Numeric> {
    pub signature_variables: Rc<SignatureVariables>,
    pub resource_variables: ResourceVariables,
    pub stage: Element,
    pub cost: T,
}

#[derive(Debug, PartialEq, Eq, Hash, Clone, Default)]
pub struct SignatureVariables {
    pub set_variables: Vec<Set>,
    pub permutation_variables: Vec<Permutation>,
    pub element_variables: Vec<Element>,
    pub integer_variables: Vec<Integer>,
    pub continuous_variables: Vec<OrderedFloat<Continuous>>,
}

#[derive(Debug, PartialEq, Clone, Default)]
pub struct ResourceVariables {
    pub integer_variables: Vec<Integer>,
    pub continuous_variables: Vec<Continuous>,
}

impl<T: Numeric> State<T> {
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
            let mut set = Set::with_capacity(metadata.set_variable_capacity_by_name(name));
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
        let mut integer_variables = Vec::with_capacity(metadata.integer_variable_names.len());
        for name in &metadata.integer_variable_names {
            let value = yaml_util::get_numeric_by_key(value, name)?;
            integer_variables.push(value);
        }
        let mut continuous_variables = Vec::with_capacity(metadata.continuous_variable_names.len());
        for name in &metadata.continuous_variable_names {
            let value = OrderedFloat(yaml_util::get_numeric_by_key(value, name)?);
            continuous_variables.push(value);
        }
        let mut integer_resource_variables =
            Vec::with_capacity(metadata.integer_resource_variable_names.len());
        for name in &metadata.integer_resource_variable_names {
            let value = yaml_util::get_numeric_by_key(value, name)?;
            integer_resource_variables.push(value);
        }
        let mut continuous_resource_variables =
            Vec::with_capacity(metadata.continuous_resource_variable_names.len());
        for name in &metadata.continuous_resource_variable_names {
            let value = yaml_util::get_numeric_by_key(value, name)?;
            continuous_resource_variables.push(value);
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
                integer_variables,
                continuous_variables,
            }),
            resource_variables: ResourceVariables {
                integer_variables: integer_resource_variables,
                continuous_variables: continuous_resource_variables,
            },
            stage,
            cost,
        })
    }
}

#[derive(Debug, PartialEq, Clone, Eq, Default)]
pub struct StateMetadata {
    pub maximize: bool,

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

    pub integer_variable_names: Vec<String>,
    pub name_to_integer_variable: collections::HashMap<String, usize>,

    pub continuous_variable_names: Vec<String>,
    pub name_to_continuous_variable: collections::HashMap<String, usize>,

    pub integer_resource_variable_names: Vec<String>,
    pub name_to_integer_resource_variable: collections::HashMap<String, usize>,
    pub integer_less_is_better: Vec<bool>,

    pub continuous_resource_variable_names: Vec<String>,
    pub name_to_continuous_resource_variable: collections::HashMap<String, usize>,
    pub continuous_less_is_better: Vec<bool>,
}

impl StateMetadata {
    pub fn number_of_objects(&self) -> usize {
        self.object_names.len()
    }

    pub fn number_of_set_variables(&self) -> usize {
        self.set_variable_names.len()
    }

    pub fn number_of_permutation_variables(&self) -> usize {
        self.permutation_variable_names.len()
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

    pub fn permutation_variable_capacity(&self, i: usize) -> usize {
        self.object_numbers[self.permutation_variable_to_object[i]]
    }

    pub fn permutation_variable_capacity_by_name(&self, name: &str) -> usize {
        self.object_numbers
            [self.permutation_variable_to_object[self.name_to_permutation_variable[name]]]
    }

    pub fn element_variable_capacity(&self, i: usize) -> usize {
        self.object_numbers[self.element_variable_to_object[i]]
    }

    pub fn element_variable_capacity_by_name(&self, name: &str) -> usize {
        self.object_numbers[self.element_variable_to_object[self.name_to_element_variable[name]]]
    }

    pub fn dominance<T: Numeric>(&self, a: &State<T>, b: &State<T>) -> Option<Ordering> {
        let status = match self.maximize {
            true => a.cost.partial_cmp(&b.cost),
            false => a.cost.partial_cmp(&b.cost).map(|x| x.reverse()),
        };
        status?;
        let status = Self::compare_resource_variables(
            &a.resource_variables.integer_variables,
            &b.resource_variables.integer_variables,
            &self.integer_less_is_better,
            status,
        );
        status?;
        Self::compare_resource_variables(
            &a.resource_variables.continuous_variables,
            &b.resource_variables.continuous_variables,
            &self.continuous_less_is_better,
            status,
        )
    }

    fn compare_resource_variables<T: Numeric>(
        a: &[T],
        b: &[T],
        less_is_better: &[bool],
        mut status: Option<Ordering>,
    ) -> Option<Ordering> {
        debug_assert!(a.len() == b.len());
        for (i, (v1, v2)) in a.iter().zip(b.iter()).enumerate() {
            match status {
                Some(Ordering::Equal) => {
                    if v1 < v2 {
                        if less_is_better[i] {
                            status = Some(Ordering::Greater);
                        } else {
                            status = Some(Ordering::Less);
                        }
                    }
                    if v1 > v2 {
                        if less_is_better[i] {
                            status = Some(Ordering::Less);
                        } else {
                            status = Some(Ordering::Greater);
                        }
                    }
                }
                Some(Ordering::Less) => {
                    if v1 < v2 {
                        if less_is_better[i] {
                            return None;
                        }
                    } else if v1 > v2 && !less_is_better[i] {
                        return None;
                    }
                }
                Some(Ordering::Greater) => {
                    if v1 > v2 {
                        if less_is_better[i] {
                            return None;
                        }
                    } else if v1 < v2 && !less_is_better[i] {
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
        let mut parameters_array: Vec<collections::HashMap<String, usize>> =
            Vec::with_capacity(array.len());
        parameters_array.push(collections::HashMap::new());
        let mut elements_in_set_variable_array: Vec<Vec<(usize, usize)>> =
            Vec::with_capacity(array.len());
        elements_in_set_variable_array.push(vec![]);
        let mut elements_in_permutation_variable_array: Vec<Vec<(usize, usize)>> =
            Vec::with_capacity(array.len());
        elements_in_permutation_variable_array.push(vec![]);
        for value in array {
            let map = yaml_util::get_map(value)?;
            let name = yaml_util::get_string_by_key(&map, "name")?;
            let object = yaml_util::get_string_by_key(&map, "object")?;
            let (n, set_index, permutation_index) =
                if let Some(i) = self.name_to_object.get(&object) {
                    (self.object_numbers[*i], None, None)
                } else if let Some(i) = self.name_to_set_variable.get(&object) {
                    (self.set_variable_capacity(*i), Some(*i), None)
                } else if let Some(i) = self.name_to_permutation_variable.get(&object) {
                    (self.permutation_variable_capacity(*i), None, Some(*i))
                } else {
                    return Err(yaml_util::YamlContentErr::new(format!(
                        "no such object, set variable, or permutation variable `{}`",
                        object
                    )));
                };
            let mut new_parameteres_set = Vec::with_capacity(parameters_array.len() * n);
            let mut new_elements_in_set_variable_array =
                Vec::with_capacity(elements_in_set_variable_array.len() * n);
            let mut new_elements_in_permutation_variable_array =
                Vec::with_capacity(elements_in_permutation_variable_array.len() * n);
            for ((parameters, elements_in_set_variable), elements_in_permutation_variable) in
                parameters_array
                    .iter()
                    .zip(elements_in_set_variable_array.iter())
                    .zip(elements_in_permutation_variable_array.iter())
            {
                for i in 0..n {
                    let mut parameters = parameters.clone();
                    parameters.insert(name.clone(), i);
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
                    new_elements_in_set_variable_array.push(elements_in_set_variable);
                    new_elements_in_permutation_variable_array
                        .push(elements_in_permutation_variable);
                }
            }
            parameters_array = new_parameteres_set;
            elements_in_set_variable_array = new_elements_in_set_variable_array;
            elements_in_permutation_variable_array = new_elements_in_permutation_variable_array;
        }

        Ok((
            parameters_array,
            elements_in_set_variable_array,
            elements_in_permutation_variable_array,
        ))
    }

    pub fn load_from_yaml(
        maximize: &yaml_rust::Yaml,
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
        let mut object_numbers: Vec<usize> = (0..object_names.len()).map(|_| 0).collect();
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
        let mut integer_variable_names = Vec::new();
        let mut name_to_integer_variable = collections::HashMap::new();
        let mut continuous_variable_names = Vec::new();
        let mut name_to_continuous_variable = collections::HashMap::new();
        let mut integer_resource_variable_names = Vec::new();
        let mut name_to_integer_resource_variable = collections::HashMap::new();
        let mut integer_less_is_better = Vec::new();
        let mut continuous_resource_variable_names = Vec::new();
        let mut name_to_continuous_resource_variable = collections::HashMap::new();
        let mut continuous_less_is_better = Vec::new();

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
                    name_to_permutation_variable
                        .insert(name.clone(), permutation_variable_names.len());
                    permutation_variable_names.push(name);
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
        let maximize = yaml_util::get_bool(maximize)?;

        Ok(StateMetadata {
            maximize,
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
    Vec<collections::HashMap<String, usize>>,
    Vec<Vec<(usize, usize)>>,
    Vec<Vec<(usize, usize)>>,
);

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn generate_metadata() -> StateMetadata {
        let object_names = vec![String::from("object"), String::from("small")];
        let object_numbers = vec![10, 2];
        let mut name_to_object = HashMap::new();
        name_to_object.insert(String::from("object"), 0);
        name_to_object.insert(String::from("small"), 1);

        let set_variable_names = vec![
            String::from("s0"),
            String::from("s1"),
            String::from("s2"),
            String::from("s3"),
        ];
        let mut name_to_set_variable = HashMap::new();
        name_to_set_variable.insert(String::from("s0"), 0);
        name_to_set_variable.insert(String::from("s1"), 1);
        name_to_set_variable.insert(String::from("s2"), 2);
        name_to_set_variable.insert(String::from("s3"), 3);
        let set_variable_to_object = vec![0, 0, 0, 1];

        let permutation_variable_names = vec![
            String::from("p0"),
            String::from("p1"),
            String::from("p2"),
            String::from("p3"),
        ];
        let mut name_to_permutation_variable = HashMap::new();
        name_to_permutation_variable.insert(String::from("p0"), 0);
        name_to_permutation_variable.insert(String::from("p1"), 1);
        name_to_permutation_variable.insert(String::from("p2"), 2);
        name_to_permutation_variable.insert(String::from("p3"), 3);
        let permutation_variable_to_object = vec![0, 0, 0, 1];

        let element_variable_names = vec![
            String::from("e0"),
            String::from("e1"),
            String::from("e2"),
            String::from("e3"),
        ];
        let mut name_to_element_variable = HashMap::new();
        name_to_element_variable.insert(String::from("e0"), 0);
        name_to_element_variable.insert(String::from("e1"), 1);
        name_to_element_variable.insert(String::from("e2"), 2);
        name_to_element_variable.insert(String::from("e3"), 3);
        let element_variable_to_object = vec![0, 0, 0, 1];

        let integer_variable_names = vec![
            String::from("i0"),
            String::from("i1"),
            String::from("i2"),
            String::from("i3"),
        ];
        let mut name_to_integer_variable = HashMap::new();
        name_to_integer_variable.insert(String::from("i0"), 0);
        name_to_integer_variable.insert(String::from("i1"), 1);
        name_to_integer_variable.insert(String::from("i2"), 2);
        name_to_integer_variable.insert(String::from("i3"), 3);

        let continuous_variable_names = vec![
            String::from("c0"),
            String::from("c1"),
            String::from("c2"),
            String::from("c3"),
        ];
        let mut name_to_continuous_variable = HashMap::new();
        name_to_continuous_variable.insert(String::from("c0"), 0);
        name_to_continuous_variable.insert(String::from("c1"), 1);
        name_to_continuous_variable.insert(String::from("c2"), 2);
        name_to_continuous_variable.insert(String::from("c3"), 3);

        let integer_resource_variable_names = vec![
            String::from("ir0"),
            String::from("ir1"),
            String::from("ir2"),
            String::from("ir3"),
        ];
        let mut name_to_integer_resource_variable = HashMap::new();
        name_to_integer_resource_variable.insert(String::from("ir0"), 0);
        name_to_integer_resource_variable.insert(String::from("ir1"), 1);
        name_to_integer_resource_variable.insert(String::from("ir2"), 2);
        name_to_integer_resource_variable.insert(String::from("ir3"), 3);

        let continuous_resource_variable_names = vec![
            String::from("cr0"),
            String::from("cr1"),
            String::from("cr2"),
            String::from("cr3"),
        ];
        let mut name_to_continuous_resource_variable = HashMap::new();
        name_to_continuous_resource_variable.insert(String::from("cr0"), 0);
        name_to_continuous_resource_variable.insert(String::from("cr1"), 1);
        name_to_continuous_resource_variable.insert(String::from("cr2"), 2);
        name_to_continuous_resource_variable.insert(String::from("cr3"), 3);

        StateMetadata {
            maximize: false,
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
            integer_variable_names,
            name_to_integer_variable,
            continuous_variable_names,
            name_to_continuous_variable,
            integer_resource_variable_names,
            name_to_integer_resource_variable,
            integer_less_is_better: vec![false, false, true, false],
            continuous_resource_variable_names,
            name_to_continuous_resource_variable,
            continuous_less_is_better: vec![false, false, true, false],
        }
    }

    #[test]
    fn state_load_from_yaml_ok() {
        let metadata = generate_metadata();

        let yaml = yaml_rust::YamlLoader::load_from_str(
            r"
s0: [0, 2]
s1: [0, 1]
s2: [0]
s3: []
p0: [0, 2]
p1: [0, 1]
p2: [0]
p3: []
e0: 0
e1: 1
e2: 2
e3: 3
i0: 0
i1: 1
i2: 2
i3: 3
c0: 0
c1: 1
c2: 2
c3: 3
ir0: 0
ir1: 1
ir2: 2
ir3: 3
cr0: 0
cr1: 1
cr2: 2
cr3: 3
",
        );
        assert!(yaml.is_ok());
        let yaml = yaml.unwrap();
        assert_eq!(yaml.len(), 1);
        let yaml = &yaml[0];
        let state = State::load_from_yaml(yaml, &metadata);
        assert!(state.is_ok());
        let mut s0 = Set::with_capacity(10);
        s0.insert(0);
        s0.insert(2);
        let mut s1 = Set::with_capacity(10);
        s1.insert(0);
        s1.insert(1);
        let mut s2 = Set::with_capacity(10);
        s2.insert(0);
        let s3 = Set::with_capacity(2);
        let mut expected = State {
            signature_variables: Rc::new(SignatureVariables {
                set_variables: vec![s0, s1, s2, s3],
                permutation_variables: vec![vec![0, 2], vec![0, 1], vec![0], vec![]],
                element_variables: vec![0, 1, 2, 3],
                integer_variables: vec![0, 1, 2, 3],
                continuous_variables: vec![
                    OrderedFloat(0.0),
                    OrderedFloat(1.0),
                    OrderedFloat(2.0),
                    OrderedFloat(3.0),
                ],
            }),
            resource_variables: ResourceVariables {
                integer_variables: vec![0, 1, 2, 3],
                continuous_variables: vec![0.0, 1.0, 2.0, 3.0],
            },
            stage: 0,
            cost: 0,
        };
        assert_eq!(state.unwrap(), expected);

        let yaml = yaml_rust::YamlLoader::load_from_str(
            r"
s0: [0, 2]
s1: [0, 1]
s2: [0]
s3: []
p0: [0, 2]
p1: [0, 1]
p2: [0]
p3: []
e0: 0
e1: 1
e2: 2
e3: 3
i0: 0
i1: 1
i2: 2
i3: 3
c0: 0
c1: 1
c2: 2
c3: 3
ir0: 0
ir1: 1
ir2: 2
ir3: 3
cr0: 0
cr1: 1
cr2: 2
cr3: 3
stage: 1
cost: 1
",
        );
        assert!(yaml.is_ok());
        let yaml = yaml.unwrap();
        assert_eq!(yaml.len(), 1);
        let yaml = &yaml[0];
        let state = State::load_from_yaml(yaml, &metadata);
        assert!(state.is_ok());
        expected.cost = 1;
        expected.stage = 1;
        assert_eq!(state.unwrap(), expected);
    }

    #[test]
    fn state_load_from_yaml_err() {
        let metadata = generate_metadata();
        let yaml = yaml_rust::YamlLoader::load_from_str(
            r"
s0: [0, 2]
s1: [0, 1]
s2: [0]
s3: []
p0: [0, 2]
p1: [0, 1]
p2: [0]
p3: []
e0: 0
e1: 1
e3: 3
i0: 0
i1: 1
i2: 2
i3: 3
c0: 0
c1: 1
c2: 2
c3: 3
ir0: 0
ir1: 1
ir2: 2
ir3: 3
cr0: 0
cr1: 1
cr2: 2
cr3: 3
",
        );
        assert!(yaml.is_ok());
        let yaml = yaml.unwrap();
        assert_eq!(yaml.len(), 1);
        let yaml = &yaml[0];
        let state = State::<Integer>::load_from_yaml(yaml, &metadata);
        assert!(state.is_err());
    }

    #[test]
    fn number_of_objects() {
        let metadata = generate_metadata();
        assert_eq!(metadata.number_of_objects(), 2);
    }

    #[test]
    fn number_of_set_variables() {
        let metadata = generate_metadata();
        assert_eq!(metadata.number_of_set_variables(), 4);
    }

    #[test]
    fn number_of_permutation_variables() {
        let metadata = generate_metadata();
        assert_eq!(metadata.number_of_permutation_variables(), 4);
    }

    #[test]
    fn number_of_element_variables() {
        let metadata = generate_metadata();
        assert_eq!(metadata.number_of_element_variables(), 4);
    }

    #[test]
    fn number_of_integer_variables() {
        let metadata = generate_metadata();
        assert_eq!(metadata.number_of_integer_variables(), 4);
    }

    #[test]
    fn number_of_continuous_variables() {
        let metadata = generate_metadata();
        assert_eq!(metadata.number_of_continuous_variables(), 4);
    }

    #[test]
    fn number_of_integer_resource_variables() {
        let metadata = generate_metadata();
        assert_eq!(metadata.number_of_integer_resource_variables(), 4);
    }

    #[test]
    fn number_of_continuous_resource_variables() {
        let metadata = generate_metadata();
        assert_eq!(metadata.number_of_continuous_resource_variables(), 4);
    }

    #[test]
    fn set_variable_capacity() {
        let metadata = generate_metadata();
        assert_eq!(metadata.set_variable_capacity(0), 10);
        assert_eq!(metadata.set_variable_capacity(1), 10);
        assert_eq!(metadata.set_variable_capacity(2), 10);
        assert_eq!(metadata.set_variable_capacity(3), 2);
    }

    #[test]
    fn set_variable_capacity_by_name() {
        let metadata = generate_metadata();
        assert_eq!(metadata.set_variable_capacity_by_name("s0"), 10);
        assert_eq!(metadata.set_variable_capacity_by_name("s1"), 10);
        assert_eq!(metadata.set_variable_capacity_by_name("s2"), 10);
        assert_eq!(metadata.set_variable_capacity_by_name("s3"), 2);
    }

    #[test]
    fn permutation_variable_capacity() {
        let metadata = generate_metadata();
        assert_eq!(metadata.permutation_variable_capacity(0), 10);
        assert_eq!(metadata.permutation_variable_capacity(1), 10);
        assert_eq!(metadata.permutation_variable_capacity(2), 10);
        assert_eq!(metadata.permutation_variable_capacity(3), 2);
    }

    #[test]
    fn permutation_variable_capacity_by_name() {
        let metadata = generate_metadata();
        assert_eq!(metadata.permutation_variable_capacity_by_name("p0"), 10);
        assert_eq!(metadata.permutation_variable_capacity_by_name("p1"), 10);
        assert_eq!(metadata.permutation_variable_capacity_by_name("p2"), 10);
        assert_eq!(metadata.permutation_variable_capacity_by_name("p3"), 2);
    }

    #[test]
    fn element_variable_capacity() {
        let metadata = generate_metadata();
        assert_eq!(metadata.element_variable_capacity(0), 10);
        assert_eq!(metadata.element_variable_capacity(1), 10);
        assert_eq!(metadata.element_variable_capacity(2), 10);
        assert_eq!(metadata.element_variable_capacity(3), 2);
    }

    #[test]
    fn element_variable_capacity_by_name() {
        let metadata = generate_metadata();
        assert_eq!(metadata.element_variable_capacity_by_name("e0"), 10);
        assert_eq!(metadata.element_variable_capacity_by_name("e1"), 10);
        assert_eq!(metadata.element_variable_capacity_by_name("e2"), 10);
        assert_eq!(metadata.element_variable_capacity_by_name("e3"), 2);
    }

    #[test]
    fn dominance() {
        let metadata = generate_metadata();

        let a = State {
            resource_variables: ResourceVariables {
                integer_variables: vec![1, 2, 2, 0],
                continuous_variables: vec![],
            },
            cost: 0,
            ..Default::default()
        };
        let b = State {
            resource_variables: ResourceVariables {
                integer_variables: vec![1, 2, 2, 0],
                continuous_variables: vec![],
            },
            cost: 0,
            ..Default::default()
        };
        assert_eq!(metadata.dominance(&a, &b), Some(Ordering::Equal));

        let b = State {
            resource_variables: ResourceVariables {
                integer_variables: vec![1, 1, 3, 0],
                continuous_variables: vec![],
            },
            cost: 0,
            ..Default::default()
        };
        assert_eq!(metadata.dominance(&a, &b), Some(Ordering::Greater));
        assert_eq!(metadata.dominance(&b, &a), Some(Ordering::Less));

        let b = State {
            resource_variables: ResourceVariables {
                integer_variables: vec![1, 3, 3, 0],
                continuous_variables: vec![],
            },
            cost: 0,
            ..Default::default()
        };
        assert!(metadata.dominance(&b, &a).is_none());

        let b = State {
            resource_variables: ResourceVariables {
                integer_variables: vec![1, 2, 2, 0],
                continuous_variables: vec![],
            },
            cost: 1,
            ..Default::default()
        };
        assert_eq!(metadata.dominance(&a, &b), Some(Ordering::Greater));
        assert_eq!(metadata.dominance(&b, &a), Some(Ordering::Less));

        let b = State {
            resource_variables: ResourceVariables {
                integer_variables: vec![1, 3, 1, 0],
                continuous_variables: vec![],
            },
            cost: 1,
            ..Default::default()
        };
        assert!(metadata.dominance(&a, &b).is_none());

        let a = State {
            resource_variables: ResourceVariables {
                integer_variables: vec![1, 2, 2, 0],
                continuous_variables: vec![1.0, 2.0, 2.0, 0.0],
            },
            cost: 0,
            ..Default::default()
        };
        let b = State {
            resource_variables: ResourceVariables {
                integer_variables: vec![1, 2, 2, 0],
                continuous_variables: vec![1.0, 1.0, 3.0, 0.0],
            },
            cost: 0,
            ..Default::default()
        };
        assert_eq!(metadata.dominance(&a, &b), Some(Ordering::Greater));
        assert_eq!(metadata.dominance(&b, &a), Some(Ordering::Less));

        let b = State {
            resource_variables: ResourceVariables {
                integer_variables: vec![1, 2, 2, 0],
                continuous_variables: vec![1.0, 3.0, 1.0, 0.0],
            },
            cost: 0,
            ..Default::default()
        };
        assert!(metadata.dominance(&a, &b).is_none());
    }

    #[test]
    #[should_panic]
    fn dominance_integer_length_panic() {
        let metadata = generate_metadata();
        let a = State {
            resource_variables: ResourceVariables {
                integer_variables: vec![1, 2, 2],
                continuous_variables: vec![],
            },
            cost: 0,
            ..Default::default()
        };
        let b = State {
            resource_variables: ResourceVariables {
                integer_variables: vec![1, 2, 2, 0],
                continuous_variables: vec![],
            },
            cost: 0,
            ..Default::default()
        };
        metadata.dominance(&b, &a);
    }

    #[test]
    #[should_panic]
    fn dominance_continuous_length_panic() {
        let metadata = generate_metadata();
        let a = State {
            resource_variables: ResourceVariables {
                integer_variables: vec![1, 2, 2, 0],
                continuous_variables: vec![1.0, 2.0, 2.0, 0.0],
            },
            cost: 0,
            ..Default::default()
        };
        let b = State {
            resource_variables: ResourceVariables {
                integer_variables: vec![1, 2, 2, 0],
                continuous_variables: vec![1.0, 1.0, 3.0],
            },
            cost: 0,
            ..Default::default()
        };
        metadata.dominance(&b, &a);
    }

    #[test]
    fn ground_parameters_from_yaml_ok() {
        let metadata = generate_metadata();
        let mut map1 = HashMap::new();
        map1.insert(String::from("v0"), 0);
        map1.insert(String::from("v1"), 0);
        let mut map2 = HashMap::new();
        map2.insert(String::from("v0"), 0);
        map2.insert(String::from("v1"), 1);
        let mut map3 = HashMap::new();
        map3.insert(String::from("v0"), 1);
        map3.insert(String::from("v1"), 0);
        let mut map4 = HashMap::new();
        map4.insert(String::from("v0"), 1);
        map4.insert(String::from("v1"), 1);
        let expected_parameters = vec![map1, map2, map3, map4];

        let yaml = yaml_rust::YamlLoader::load_from_str(
            r"
- name: v0
  object: small
- name: v1
  object: s3
",
        );
        assert!(yaml.is_ok());
        let yaml = yaml.unwrap();
        assert_eq!(yaml.len(), 1);
        let yaml = &yaml[0];
        let result = metadata.ground_parameters_from_yaml(&yaml);
        assert!(result.is_ok());
        let (parameters, elements_in_set_variable_array, elements_in_permutation_variable_array) =
            result.unwrap();
        let expected_elements_in_set_variable_array =
            vec![vec![(3, 0)], vec![(3, 1)], vec![(3, 0)], vec![(3, 1)]];
        let expected_elements_in_permutation_variable_array = vec![vec![], vec![], vec![], vec![]];
        assert_eq!(parameters, expected_parameters);
        assert_eq!(
            elements_in_set_variable_array,
            expected_elements_in_set_variable_array
        );
        assert_eq!(
            elements_in_permutation_variable_array,
            expected_elements_in_permutation_variable_array
        );

        let yaml = yaml_rust::YamlLoader::load_from_str(
            r"
- name: v0
  object: s3
- name: v1
  object: p3
",
        );
        assert!(yaml.is_ok());
        let yaml = yaml.unwrap();
        assert_eq!(yaml.len(), 1);
        let yaml = &yaml[0];
        let result = metadata.ground_parameters_from_yaml(&yaml);
        assert!(result.is_ok());
        let (parameters, elements_in_set_variable_array, elements_in_permutation_variable_array) =
            result.unwrap();
        let expected_elements_in_set_variable_array =
            vec![vec![(3, 0)], vec![(3, 0)], vec![(3, 1)], vec![(3, 1)]];
        let expected_elements_in_permutation_variable_array =
            vec![vec![(3, 0)], vec![(3, 1)], vec![(3, 0)], vec![(3, 1)]];
        assert_eq!(parameters, expected_parameters);
        assert_eq!(
            elements_in_set_variable_array,
            expected_elements_in_set_variable_array
        );
        assert_eq!(
            elements_in_permutation_variable_array,
            expected_elements_in_permutation_variable_array
        );
    }

    #[test]
    fn ground_parameters_from_yaml_err() {
        let metadata = generate_metadata();
        let yaml = yaml_rust::YamlLoader::load_from_str(
            r"
- name: v0
  object: small
- name: v1
  object: s5
",
        );
        assert!(yaml.is_ok());
        let yaml = yaml.unwrap();
        assert_eq!(yaml.len(), 1);
        let yaml = &yaml[0];
        let result = metadata.ground_parameters_from_yaml(&yaml);
        assert!(result.is_err());
    }

    #[test]
    fn state_metadata_load_from_yaml_ok() {
        let expected = generate_metadata();
        let maximize = yaml_rust::YamlLoader::load_from_str("false");
        assert!(maximize.is_ok());
        let maximize = maximize.unwrap();
        assert_eq!(maximize.len(), 1);
        let maximize = &maximize[0];

        let objects = r"
- object
- small
";
        let variables = r"
- name: s0
  type: set
  object: object
- name: s1
  type: set
  object: object
- name: s2
  type: set
  object: object
- name: s3
  type: set
  object: small
- name: p0
  type: permutation
  object: object
- name: p1
  type: permutation 
  object: object
- name: p2
  type: permutation
  object: object
- name: p3
  type: permutation
  object: small
- name: e0
  type: element 
  object: object
- name: e1
  type: element
  object: object
- name: e2
  type: element
  object: object
- name: e3
  type: element
  object: small
- name: i0
  type: integer
- name: i1
  type: integer
- name: i2
  type: integer 
- name: i3
  type: integer
- name: c0
  type: continuous 
- name: c1
  type: continuous
- name: c2
  type: continuous
- name: c3
  type: continuous
- name: ir0
  type: integer
  preference: greater
- name: ir1
  type: integer
  preference: greater
- name: ir2
  type: integer 
  preference: less
- name: ir3
  type: integer
  preference: greater
- name: cr0
  type: continuous 
  preference: greater
- name: cr1
  type: continuous
  preference: greater
- name: cr2
  type: continuous
  preference: less
- name: cr3
  type: continuous
  preference: greater
";
        let object_numbers = r"
object: 10
small: 2
";
        let objects = yaml_rust::YamlLoader::load_from_str(objects);
        assert!(objects.is_ok());
        let objects = objects.unwrap();
        assert_eq!(objects.len(), 1);
        let objects = &objects[0];

        let variables = yaml_rust::YamlLoader::load_from_str(variables);
        assert!(variables.is_ok());
        let variables = variables.unwrap();
        assert_eq!(variables.len(), 1);
        let variables = &variables[0];

        let object_numbers = yaml_rust::YamlLoader::load_from_str(object_numbers);
        assert!(object_numbers.is_ok());
        let object_numbers = object_numbers.unwrap();
        assert_eq!(object_numbers.len(), 1);
        let object_numbers = &object_numbers[0];

        let metadata = StateMetadata::load_from_yaml(maximize, objects, variables, object_numbers);
        assert!(metadata.is_ok());
        assert_eq!(metadata.unwrap(), expected);
    }

    #[test]
    fn state_metadata_load_from_yaml_err() {
        let maximize = yaml_rust::YamlLoader::load_from_str("false");
        assert!(maximize.is_ok());
        let maximize = maximize.unwrap();
        assert_eq!(maximize.len(), 1);
        let maximize = &maximize[0];

        let objects = r"
- object
- small
";
        let variables = r"
- name: s0
  type: null
  object: small 
";
        let object_numbers = r"
object: 10
small: 2
";
        let objects = yaml_rust::YamlLoader::load_from_str(objects);
        assert!(objects.is_ok());
        let objects = objects.unwrap();
        assert_eq!(objects.len(), 1);
        let objects = &objects[0];

        let variables = yaml_rust::YamlLoader::load_from_str(variables);
        assert!(variables.is_ok());
        let variables = variables.unwrap();
        assert_eq!(variables.len(), 1);
        let variables = &variables[0];

        let object_numbers = yaml_rust::YamlLoader::load_from_str(object_numbers);
        assert!(object_numbers.is_ok());
        let object_numbers = object_numbers.unwrap();
        assert_eq!(object_numbers.len(), 1);
        let object_numbers = &object_numbers[0];

        let metadata = StateMetadata::load_from_yaml(maximize, objects, variables, object_numbers);
        assert!(metadata.is_err());

        let variables = r"
- name: s0
  type: set
  object: null
";
        let variables = yaml_rust::YamlLoader::load_from_str(variables);
        assert!(variables.is_ok());
        let variables = variables.unwrap();
        assert_eq!(variables.len(), 1);
        let variables = &variables[0];
        let metadata = StateMetadata::load_from_yaml(maximize, objects, variables, object_numbers);
        assert!(metadata.is_err());

        let variables = r"
- object: object
  type: set
";
        let variables = yaml_rust::YamlLoader::load_from_str(variables);
        assert!(variables.is_ok());
        let variables = variables.unwrap();
        assert_eq!(variables.len(), 1);
        let variables = &variables[0];
        let metadata = StateMetadata::load_from_yaml(maximize, objects, variables, object_numbers);
        assert!(metadata.is_err());

        let variables = r"
- name: s0
  type: set
";
        let variables = yaml_rust::YamlLoader::load_from_str(variables);
        assert!(variables.is_ok());
        let variables = variables.unwrap();
        assert_eq!(variables.len(), 1);
        let variables = &variables[0];
        let metadata = StateMetadata::load_from_yaml(maximize, objects, variables, object_numbers);
        assert!(metadata.is_err());

        let variables = r"
- name: s0
  object: object
";
        let variables = yaml_rust::YamlLoader::load_from_str(variables);
        assert!(variables.is_ok());
        let variables = variables.unwrap();
        assert_eq!(variables.len(), 1);
        let variables = &variables[0];
        let metadata = StateMetadata::load_from_yaml(maximize, objects, variables, object_numbers);
        assert!(metadata.is_err());

        let variables = r"
- name: s0
  type: set
  object: object
";
        let variables = yaml_rust::YamlLoader::load_from_str(variables);
        assert!(variables.is_ok());
        let variables = variables.unwrap();
        assert_eq!(variables.len(), 1);
        let variables = &variables[0];

        let object_numbers = r"
object: 10
";
        let object_numbers = yaml_rust::YamlLoader::load_from_str(object_numbers);
        assert!(object_numbers.is_ok());
        let object_numbers = object_numbers.unwrap();
        assert_eq!(object_numbers.len(), 1);
        let object_numbers = &object_numbers[0];
        let metadata = StateMetadata::load_from_yaml(maximize, objects, variables, object_numbers);
        assert!(metadata.is_err());

        let object_numbers = r"
object: 10
small: 2
";
        let object_numbers = yaml_rust::YamlLoader::load_from_str(object_numbers);
        assert!(object_numbers.is_ok());
        let object_numbers = object_numbers.unwrap();
        assert_eq!(object_numbers.len(), 1);
        let object_numbers = &object_numbers[0];

        let maximize = yaml_rust::YamlLoader::load_from_str("null");
        assert!(maximize.is_ok());
        let maximize = maximize.unwrap();
        assert_eq!(maximize.len(), 1);
        let maximize = &maximize[0];
        let metadata = StateMetadata::load_from_yaml(maximize, objects, variables, object_numbers);
        assert!(metadata.is_err());
    }
}
