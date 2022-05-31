use crate::effect;
use crate::table_registry;
use crate::variable::{Continuous, Element, Integer, Numeric, Set, Vector};
use crate::yaml_util;
use lazy_static::lazy_static;
use rustc_hash::{FxHashMap, FxHashSet};
use std::cmp::Ordering;
use std::fmt;

pub trait DPState: Clone + fmt::Debug {
    fn get_set_variable(&self, i: usize) -> &Set;
    fn get_vector_variable(&self, i: usize) -> &Vector;
    fn get_element_variable(&self, i: usize) -> Element;
    fn get_integer_variable(&self, i: usize) -> Integer;
    fn get_continuous_variable(&self, i: usize) -> Continuous;
    fn get_integer_resource_variable(&self, i: usize) -> Integer;
    fn get_continuous_resource_variable(&self, i: usize) -> Continuous;
    fn apply_effect(
        &self,
        effect: &effect::Effect,
        registry: &table_registry::TableRegistry,
    ) -> Self;
    fn apply_effect_in_place(
        &mut self,
        effect: &effect::Effect,
        registry: &table_registry::TableRegistry,
    );
}

#[derive(Debug, PartialEq, Clone, Default)]
pub struct SignatureVariables {
    pub set_variables: Vec<Set>,
    pub vector_variables: Vec<Vector>,
    pub element_variables: Vec<Element>,
    pub integer_variables: Vec<Integer>,
    pub continuous_variables: Vec<Continuous>,
}

#[derive(Debug, PartialEq, Clone, Default)]
pub struct ResourceVariables {
    pub integer_variables: Vec<Integer>,
    pub continuous_variables: Vec<Continuous>,
}

#[derive(Debug, PartialEq, Clone, Default)]
pub struct State {
    pub signature_variables: SignatureVariables,
    pub resource_variables: ResourceVariables,
}

impl DPState for State {
    #[inline]
    fn get_set_variable(&self, i: usize) -> &Set {
        &self.signature_variables.set_variables[i]
    }

    #[inline]
    fn get_vector_variable(&self, i: usize) -> &Vector {
        &self.signature_variables.vector_variables[i]
    }

    #[inline]
    fn get_element_variable(&self, i: usize) -> Element {
        self.signature_variables.element_variables[i]
    }

    #[inline]
    fn get_integer_variable(&self, i: usize) -> Integer {
        self.signature_variables.integer_variables[i]
    }

    #[inline]
    fn get_continuous_variable(&self, i: usize) -> Continuous {
        self.signature_variables.continuous_variables[i]
    }

    #[inline]
    fn get_integer_resource_variable(&self, i: usize) -> Integer {
        self.resource_variables.integer_variables[i]
    }

    #[inline]
    fn get_continuous_resource_variable(&self, i: usize) -> Continuous {
        self.resource_variables.continuous_variables[i]
    }

    fn apply_effect(
        &self,
        effect: &effect::Effect,
        registry: &table_registry::TableRegistry,
    ) -> Self {
        let len = self.signature_variables.set_variables.len();
        let mut set_variables = Vec::with_capacity(len);
        let mut i = 0;
        for e in &effect.set_effects {
            while i < e.0 {
                set_variables.push(self.signature_variables.set_variables[i].clone());
                i += 1;
            }
            set_variables.push(e.1.eval(self, registry));
            i += 1;
        }
        while i < len {
            set_variables.push(self.signature_variables.set_variables[i].clone());
            i += 1;
        }

        let len = self.signature_variables.vector_variables.len();
        let mut vector_variables = Vec::with_capacity(len);
        for e in &effect.vector_effects {
            while i < e.0 {
                vector_variables.push(self.signature_variables.vector_variables[i].clone());
                i += 1;
            }
            vector_variables.push(e.1.eval(self, registry));
            i += 1;
        }
        while i < len {
            vector_variables.push(self.signature_variables.vector_variables[i].clone());
            i += 1;
        }

        let mut element_variables = self.signature_variables.element_variables.clone();
        for e in &effect.element_effects {
            element_variables[e.0] = e.1.eval(self, registry);
        }

        let mut integer_variables = self.signature_variables.integer_variables.clone();
        for e in &effect.integer_effects {
            integer_variables[e.0] = e.1.eval(self, registry);
        }

        let mut continuous_variables = self.signature_variables.continuous_variables.clone();
        for e in &effect.continuous_effects {
            continuous_variables[e.0] = e.1.eval(self, registry);
        }

        let mut integer_resource_variables = self.resource_variables.integer_variables.clone();
        for e in &effect.integer_resource_effects {
            integer_resource_variables[e.0] = e.1.eval(self, registry);
        }

        let mut continuous_resource_variables =
            self.resource_variables.continuous_variables.clone();
        for e in &effect.continuous_resource_effects {
            continuous_resource_variables[e.0] = e.1.eval(self, registry);
        }

        State {
            signature_variables: SignatureVariables {
                set_variables,
                vector_variables,
                element_variables,
                integer_variables,
                continuous_variables,
            },
            resource_variables: ResourceVariables {
                integer_variables: integer_resource_variables,
                continuous_variables: continuous_resource_variables,
            },
        }
    }

    fn apply_effect_in_place(
        &mut self,
        effect: &effect::Effect,
        registry: &table_registry::TableRegistry,
    ) {
        for e in &effect.set_effects {
            self.signature_variables.set_variables[e.0] = e.1.eval(self, registry);
        }
        for e in &effect.vector_effects {
            self.signature_variables.vector_variables[e.0] = e.1.eval(self, registry);
        }
        for e in &effect.element_effects {
            self.signature_variables.element_variables[e.0] = e.1.eval(self, registry);
        }
        for e in &effect.integer_effects {
            self.signature_variables.integer_variables[e.0] = e.1.eval(self, registry);
        }
        for e in &effect.continuous_effects {
            self.signature_variables.continuous_variables[e.0] = e.1.eval(self, registry);
        }
        for e in &effect.integer_resource_effects {
            self.resource_variables.integer_variables[e.0] = e.1.eval(self, registry);
        }
        for e in &effect.continuous_resource_effects {
            self.resource_variables.continuous_variables[e.0] = e.1.eval(self, registry);
        }
    }
}

impl State {
    pub fn load_from_yaml(
        value: &yaml_rust::Yaml,
        metadata: &StateMetadata,
    ) -> Result<State, yaml_util::YamlContentErr> {
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
        let mut vector_variables = Vec::with_capacity(metadata.vector_variable_names.len());
        for name in &metadata.vector_variable_names {
            let vector = yaml_util::get_usize_array_by_key(value, name)?;
            vector_variables.push(vector);
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
            let value = yaml_util::get_numeric_by_key(value, name)?;
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
        Ok(State {
            signature_variables: SignatureVariables {
                set_variables,
                vector_variables,
                element_variables,
                integer_variables,
                continuous_variables,
            },
            resource_variables: ResourceVariables {
                integer_variables: integer_resource_variables,
                continuous_variables: continuous_resource_variables,
            },
        })
    }
}

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
            let name = yaml_util::get_string_by_key(map, "name")?;
            if let Some(name) = reserved_names.get(&name) {
                return Err(yaml_util::YamlContentErr::new(format!(
                    "parameter name `{}` is already used",
                    name
                )));
            }
            reserved_names.insert(name.clone());
            let object = yaml_util::get_string_by_key(map, "object")?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expression::*;
    use crate::table;
    use crate::table_data;

    fn generate_registry() -> table_registry::TableRegistry {
        let tables_1d = vec![table::Table1D::new(vec![10, 20, 30])];
        let mut name_to_table_1d = FxHashMap::default();
        name_to_table_1d.insert(String::from("f1"), 0);

        let tables_2d = vec![table::Table2D::new(vec![
            vec![10, 20, 30],
            vec![40, 50, 60],
        ])];
        let mut name_to_table_2d = FxHashMap::default();
        name_to_table_2d.insert(String::from("f2"), 0);

        table_registry::TableRegistry {
            integer_tables: table_data::TableData {
                tables_1d,
                name_to_table_1d,
                tables_2d,
                name_to_table_2d,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn generate_metadata() -> StateMetadata {
        let object_names = vec![String::from("object"), String::from("small")];
        let object_numbers = vec![10, 2];
        let mut name_to_object = FxHashMap::default();
        name_to_object.insert(String::from("object"), 0);
        name_to_object.insert(String::from("small"), 1);

        let set_variable_names = vec![
            String::from("s0"),
            String::from("s1"),
            String::from("s2"),
            String::from("s3"),
        ];
        let mut name_to_set_variable = FxHashMap::default();
        name_to_set_variable.insert(String::from("s0"), 0);
        name_to_set_variable.insert(String::from("s1"), 1);
        name_to_set_variable.insert(String::from("s2"), 2);
        name_to_set_variable.insert(String::from("s3"), 3);
        let set_variable_to_object = vec![0, 0, 0, 1];

        let vector_variable_names = vec![
            String::from("p0"),
            String::from("p1"),
            String::from("p2"),
            String::from("p3"),
        ];
        let mut name_to_vector_variable = FxHashMap::default();
        name_to_vector_variable.insert(String::from("p0"), 0);
        name_to_vector_variable.insert(String::from("p1"), 1);
        name_to_vector_variable.insert(String::from("p2"), 2);
        name_to_vector_variable.insert(String::from("p3"), 3);
        let vector_variable_to_object = vec![0, 0, 0, 1];

        let element_variable_names = vec![
            String::from("e0"),
            String::from("e1"),
            String::from("e2"),
            String::from("e3"),
        ];
        let mut name_to_element_variable = FxHashMap::default();
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
        let mut name_to_integer_variable = FxHashMap::default();
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
        let mut name_to_continuous_variable = FxHashMap::default();
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
        let mut name_to_integer_resource_variable = FxHashMap::default();
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
        let mut name_to_continuous_resource_variable = FxHashMap::default();
        name_to_continuous_resource_variable.insert(String::from("cr0"), 0);
        name_to_continuous_resource_variable.insert(String::from("cr1"), 1);
        name_to_continuous_resource_variable.insert(String::from("cr2"), 2);
        name_to_continuous_resource_variable.insert(String::from("cr3"), 3);

        StateMetadata {
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
        let expected = State {
            signature_variables: SignatureVariables {
                set_variables: vec![s0, s1, s2, s3],
                vector_variables: vec![vec![0, 2], vec![0, 1], vec![0], vec![]],
                element_variables: vec![0, 1, 2, 3],
                integer_variables: vec![0, 1, 2, 3],
                continuous_variables: vec![0.0, 1.0, 2.0, 3.0],
            },
            resource_variables: ResourceVariables {
                integer_variables: vec![0, 1, 2, 3],
                continuous_variables: vec![0.0, 1.0, 2.0, 3.0],
            },
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
",
        );
        assert!(yaml.is_ok());
        let yaml = yaml.unwrap();
        assert_eq!(yaml.len(), 1);
        let yaml = &yaml[0];
        let state = State::load_from_yaml(yaml, &metadata);
        assert!(state.is_ok());
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
        let state = State::load_from_yaml(yaml, &metadata);
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
    fn number_of_vector_variables() {
        let metadata = generate_metadata();
        assert_eq!(metadata.number_of_vector_variables(), 4);
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
    fn vector_variable_capacity() {
        let metadata = generate_metadata();
        assert_eq!(metadata.vector_variable_capacity(0), 10);
        assert_eq!(metadata.vector_variable_capacity(1), 10);
        assert_eq!(metadata.vector_variable_capacity(2), 10);
        assert_eq!(metadata.vector_variable_capacity(3), 2);
    }

    #[test]
    fn vector_variable_capacity_by_name() {
        let metadata = generate_metadata();
        assert_eq!(metadata.vector_variable_capacity_by_name("p0"), 10);
        assert_eq!(metadata.vector_variable_capacity_by_name("p1"), 10);
        assert_eq!(metadata.vector_variable_capacity_by_name("p2"), 10);
        assert_eq!(metadata.vector_variable_capacity_by_name("p3"), 2);
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
    fn get_name_set() {
        let metadata = generate_metadata();
        let mut expected = FxHashSet::default();
        expected.insert(String::from("object"));
        expected.insert(String::from("small"));
        expected.insert(String::from("s0"));
        expected.insert(String::from("s1"));
        expected.insert(String::from("s2"));
        expected.insert(String::from("s3"));
        expected.insert(String::from("p0"));
        expected.insert(String::from("p1"));
        expected.insert(String::from("p2"));
        expected.insert(String::from("p3"));
        expected.insert(String::from("e0"));
        expected.insert(String::from("e1"));
        expected.insert(String::from("e2"));
        expected.insert(String::from("e3"));
        expected.insert(String::from("i0"));
        expected.insert(String::from("i1"));
        expected.insert(String::from("i2"));
        expected.insert(String::from("i3"));
        expected.insert(String::from("c0"));
        expected.insert(String::from("c1"));
        expected.insert(String::from("c2"));
        expected.insert(String::from("c3"));
        expected.insert(String::from("ir0"));
        expected.insert(String::from("ir1"));
        expected.insert(String::from("ir2"));
        expected.insert(String::from("ir3"));
        expected.insert(String::from("cr0"));
        expected.insert(String::from("cr1"));
        expected.insert(String::from("cr2"));
        expected.insert(String::from("cr3"));
        assert_eq!(metadata.get_name_set(), expected);
    }

    #[test]
    fn state_getter() {
        let mut set1 = Set::with_capacity(3);
        set1.insert(0);
        set1.insert(2);
        let mut set2 = Set::with_capacity(3);
        set2.insert(0);
        set2.insert(1);
        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![set1.clone(), set2.clone()],
                vector_variables: vec![vec![0, 2], vec![1, 2]],
                element_variables: vec![1, 2],
                integer_variables: vec![1, 2, 3],
                continuous_variables: vec![1.0, 2.0, 3.0],
            },
            resource_variables: ResourceVariables {
                integer_variables: vec![4, 5, 6],
                continuous_variables: vec![4.0, 5.0, 6.0],
            },
        };
        assert_eq!(state.get_set_variable(0), &set1);
        assert_eq!(state.get_set_variable(1), &set2);
        assert_eq!(state.get_vector_variable(0), &vec![0, 2]);
        assert_eq!(state.get_vector_variable(1), &vec![1, 2]);
        assert_eq!(state.get_element_variable(0), 1);
        assert_eq!(state.get_element_variable(1), 2);
        assert_eq!(state.get_integer_variable(0), 1);
        assert_eq!(state.get_integer_variable(1), 2);
        assert_eq!(state.get_integer_variable(2), 3);
        assert_eq!(state.get_continuous_variable(0), 1.0);
        assert_eq!(state.get_continuous_variable(1), 2.0);
        assert_eq!(state.get_continuous_variable(2), 3.0);
        assert_eq!(state.get_integer_resource_variable(0), 4);
        assert_eq!(state.get_integer_resource_variable(1), 5);
        assert_eq!(state.get_integer_resource_variable(2), 6);
        assert_eq!(state.get_continuous_resource_variable(0), 4.0);
        assert_eq!(state.get_continuous_resource_variable(1), 5.0);
        assert_eq!(state.get_continuous_resource_variable(2), 6.0);
    }

    #[test]
    fn appy_effect() {
        let mut set1 = Set::with_capacity(3);
        set1.insert(0);
        set1.insert(2);
        let mut set2 = Set::with_capacity(3);
        set2.insert(0);
        set2.insert(1);
        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![set1, set2],
                vector_variables: vec![vec![0, 2], vec![1, 2]],
                element_variables: vec![1, 2],
                integer_variables: vec![1, 2, 3],
                continuous_variables: vec![1.0, 2.0, 3.0],
            },
            resource_variables: ResourceVariables {
                integer_variables: vec![4, 5, 6],
                continuous_variables: vec![4.0, 5.0, 6.0],
            },
        };
        let registry = generate_registry();
        let set_effect1 = SetExpression::SetElementOperation(
            SetElementOperator::Add,
            ElementExpression::Constant(1),
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
        );
        let set_effect2 = SetExpression::SetElementOperation(
            SetElementOperator::Remove,
            ElementExpression::Constant(0),
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(1))),
        );
        let vector_effect1 = VectorExpression::Push(
            ElementExpression::Constant(1),
            Box::new(VectorExpression::Reference(ReferenceExpression::Variable(
                0,
            ))),
        );
        let vector_effect2 = VectorExpression::Push(
            ElementExpression::Constant(0),
            Box::new(VectorExpression::Reference(ReferenceExpression::Variable(
                1,
            ))),
        );
        let element_effect1 = ElementExpression::Constant(2);
        let element_effect2 = ElementExpression::Constant(1);
        let integer_effect1 = NumericExpression::NumericOperation(
            NumericOperator::Subtract,
            Box::new(NumericExpression::IntegerVariable(0)),
            Box::new(NumericExpression::Constant(1)),
        );
        let integer_effect2 = NumericExpression::NumericOperation(
            NumericOperator::Multiply,
            Box::new(NumericExpression::IntegerVariable(1)),
            Box::new(NumericExpression::Constant(2)),
        );
        let continuous_effect1 = NumericExpression::NumericOperation(
            NumericOperator::Subtract,
            Box::new(NumericExpression::ContinuousVariable(0)),
            Box::new(NumericExpression::Constant(1.0)),
        );
        let continuous_effect2 = NumericExpression::NumericOperation(
            NumericOperator::Multiply,
            Box::new(NumericExpression::ContinuousVariable(1)),
            Box::new(NumericExpression::Constant(2.0)),
        );
        let integer_resource_effect1 = NumericExpression::NumericOperation(
            NumericOperator::Add,
            Box::new(NumericExpression::IntegerResourceVariable(0)),
            Box::new(NumericExpression::Constant(1)),
        );
        let integer_resource_effect2 = NumericExpression::NumericOperation(
            NumericOperator::Divide,
            Box::new(NumericExpression::IntegerResourceVariable(1)),
            Box::new(NumericExpression::Constant(2)),
        );
        let continuous_resource_effect1 = NumericExpression::NumericOperation(
            NumericOperator::Add,
            Box::new(NumericExpression::ContinuousResourceVariable(0)),
            Box::new(NumericExpression::Constant(1.0)),
        );
        let continuous_resource_effect2 = NumericExpression::NumericOperation(
            NumericOperator::Divide,
            Box::new(NumericExpression::ContinuousResourceVariable(1)),
            Box::new(NumericExpression::Constant(2.0)),
        );
        let effect = effect::Effect {
            set_effects: vec![(0, set_effect1), (1, set_effect2)],
            vector_effects: vec![(0, vector_effect1), (1, vector_effect2)],
            element_effects: vec![(0, element_effect1), (1, element_effect2)],
            integer_effects: vec![(0, integer_effect1), (1, integer_effect2)],
            continuous_effects: vec![(0, continuous_effect1), (1, continuous_effect2)],
            integer_resource_effects: vec![
                (0, integer_resource_effect1),
                (1, integer_resource_effect2),
            ],
            continuous_resource_effects: vec![
                (0, continuous_resource_effect1),
                (1, continuous_resource_effect2),
            ],
        };

        let mut set1 = Set::with_capacity(3);
        set1.insert(0);
        set1.insert(1);
        set1.insert(2);
        let mut set2 = Set::with_capacity(3);
        set2.insert(1);
        let expected = State {
            signature_variables: SignatureVariables {
                set_variables: vec![set1, set2],
                vector_variables: vec![vec![0, 2, 1], vec![1, 2, 0]],
                element_variables: vec![2, 1],
                integer_variables: vec![0, 4, 3],
                continuous_variables: vec![0.0, 4.0, 3.0],
            },
            resource_variables: ResourceVariables {
                integer_variables: vec![5, 2, 6],
                continuous_variables: vec![5.0, 2.5, 6.0],
            },
        };
        let successor = state.apply_effect(&effect, &registry);
        assert_eq!(successor, expected);
    }

    #[test]
    fn appy_effect_in_place() {
        let mut set1 = Set::with_capacity(3);
        set1.insert(0);
        set1.insert(2);
        let mut set2 = Set::with_capacity(3);
        set2.insert(0);
        set2.insert(1);
        let mut state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![set1, set2],
                vector_variables: vec![vec![0, 2], vec![1, 2]],
                element_variables: vec![1, 2],
                integer_variables: vec![1, 2, 3],
                continuous_variables: vec![1.0, 2.0, 3.0],
            },
            resource_variables: ResourceVariables {
                integer_variables: vec![4, 5, 6],
                continuous_variables: vec![4.0, 5.0, 6.0],
            },
        };
        let registry = generate_registry();
        let set_effect1 = SetExpression::SetElementOperation(
            SetElementOperator::Add,
            ElementExpression::Constant(1),
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
        );
        let set_effect2 = SetExpression::SetElementOperation(
            SetElementOperator::Remove,
            ElementExpression::Constant(0),
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(1))),
        );
        let vector_effect1 = VectorExpression::Push(
            ElementExpression::Constant(1),
            Box::new(VectorExpression::Reference(ReferenceExpression::Variable(
                0,
            ))),
        );
        let vector_effect2 = VectorExpression::Push(
            ElementExpression::Constant(0),
            Box::new(VectorExpression::Reference(ReferenceExpression::Variable(
                1,
            ))),
        );
        let element_effect1 = ElementExpression::Constant(2);
        let element_effect2 = ElementExpression::Constant(1);
        let integer_effect1 = NumericExpression::NumericOperation(
            NumericOperator::Subtract,
            Box::new(NumericExpression::IntegerVariable(0)),
            Box::new(NumericExpression::Constant(1)),
        );
        let integer_effect2 = NumericExpression::NumericOperation(
            NumericOperator::Multiply,
            Box::new(NumericExpression::IntegerVariable(1)),
            Box::new(NumericExpression::Constant(2)),
        );
        let continuous_effect1 = NumericExpression::NumericOperation(
            NumericOperator::Subtract,
            Box::new(NumericExpression::ContinuousVariable(0)),
            Box::new(NumericExpression::Constant(1.0)),
        );
        let continuous_effect2 = NumericExpression::NumericOperation(
            NumericOperator::Multiply,
            Box::new(NumericExpression::ContinuousVariable(1)),
            Box::new(NumericExpression::Constant(2.0)),
        );
        let integer_resource_effect1 = NumericExpression::NumericOperation(
            NumericOperator::Add,
            Box::new(NumericExpression::IntegerResourceVariable(0)),
            Box::new(NumericExpression::Constant(1)),
        );
        let integer_resource_effect2 = NumericExpression::NumericOperation(
            NumericOperator::Divide,
            Box::new(NumericExpression::IntegerResourceVariable(1)),
            Box::new(NumericExpression::Constant(2)),
        );
        let continuous_resource_effect1 = NumericExpression::NumericOperation(
            NumericOperator::Add,
            Box::new(NumericExpression::ContinuousResourceVariable(0)),
            Box::new(NumericExpression::Constant(1.0)),
        );
        let continuous_resource_effect2 = NumericExpression::NumericOperation(
            NumericOperator::Divide,
            Box::new(NumericExpression::ContinuousResourceVariable(1)),
            Box::new(NumericExpression::Constant(2.0)),
        );
        let effect = effect::Effect {
            set_effects: vec![(0, set_effect1), (1, set_effect2)],
            vector_effects: vec![(0, vector_effect1), (1, vector_effect2)],
            element_effects: vec![(0, element_effect1), (1, element_effect2)],
            integer_effects: vec![(0, integer_effect1), (1, integer_effect2)],
            continuous_effects: vec![(0, continuous_effect1), (1, continuous_effect2)],
            integer_resource_effects: vec![
                (0, integer_resource_effect1),
                (1, integer_resource_effect2),
            ],
            continuous_resource_effects: vec![
                (0, continuous_resource_effect1),
                (1, continuous_resource_effect2),
            ],
        };

        let mut set1 = Set::with_capacity(3);
        set1.insert(0);
        set1.insert(1);
        set1.insert(2);
        let mut set2 = Set::with_capacity(3);
        set2.insert(1);
        let expected = State {
            signature_variables: SignatureVariables {
                set_variables: vec![set1, set2],
                vector_variables: vec![vec![0, 2, 1], vec![1, 2, 0]],
                element_variables: vec![2, 1],
                integer_variables: vec![0, 4, 3],
                continuous_variables: vec![0.0, 4.0, 3.0],
            },
            resource_variables: ResourceVariables {
                integer_variables: vec![5, 2, 6],
                continuous_variables: vec![5.0, 2.5, 6.0],
            },
        };
        state.apply_effect_in_place(&effect, &registry);
        assert_eq!(state, expected);
    }

    #[test]
    fn dominance() {
        let metadata = generate_metadata();

        let a = State {
            resource_variables: ResourceVariables {
                integer_variables: vec![1, 2, 2, 0],
                continuous_variables: vec![0.0, 0.0, 0.0, 0.0],
            },
            ..Default::default()
        };
        let b = State {
            resource_variables: ResourceVariables {
                integer_variables: vec![1, 2, 2, 0],
                continuous_variables: vec![0.0, 0.0, 0.0, 0.0],
            },
            ..Default::default()
        };
        assert_eq!(metadata.dominance(&a, &b), Some(Ordering::Equal));

        let b = State {
            resource_variables: ResourceVariables {
                integer_variables: vec![1, 1, 3, 0],
                continuous_variables: vec![0.0, 0.0, 0.0, 0.0],
            },
            ..Default::default()
        };
        assert_eq!(metadata.dominance(&a, &b), Some(Ordering::Greater));
        assert_eq!(metadata.dominance(&b, &a), Some(Ordering::Less));

        let b = State {
            resource_variables: ResourceVariables {
                integer_variables: vec![1, 3, 3, 0],
                continuous_variables: vec![0.0, 0.0, 0.0, 0.0],
            },
            ..Default::default()
        };
        assert!(metadata.dominance(&b, &a).is_none());

        let a = State {
            resource_variables: ResourceVariables {
                integer_variables: vec![1, 2, 2, 0],
                continuous_variables: vec![1.0, 2.0, 2.0, 0.0],
            },
            ..Default::default()
        };
        let b = State {
            resource_variables: ResourceVariables {
                integer_variables: vec![1, 2, 2, 0],
                continuous_variables: vec![1.0, 1.0, 3.0, 0.0],
            },
            ..Default::default()
        };
        assert_eq!(metadata.dominance(&a, &b), Some(Ordering::Greater));
        assert_eq!(metadata.dominance(&b, &a), Some(Ordering::Less));

        let b = State {
            resource_variables: ResourceVariables {
                integer_variables: vec![1, 2, 2, 0],
                continuous_variables: vec![1.0, 3.0, 4.0, 0.0],
            },
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
            ..Default::default()
        };
        let b = State {
            resource_variables: ResourceVariables {
                integer_variables: vec![1, 2, 2, 0],
                continuous_variables: vec![],
            },
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
            ..Default::default()
        };
        let b = State {
            resource_variables: ResourceVariables {
                integer_variables: vec![1, 2, 2, 0],
                continuous_variables: vec![1.0, 1.0, 3.0],
            },
            ..Default::default()
        };
        metadata.dominance(&b, &a);
    }

    #[test]
    fn ground_parameters_from_yaml_ok() {
        let metadata = generate_metadata();
        let mut map1 = FxHashMap::default();
        map1.insert(String::from("v0"), 0);
        map1.insert(String::from("v1"), 0);
        let mut map2 = FxHashMap::default();
        map2.insert(String::from("v0"), 0);
        map2.insert(String::from("v1"), 1);
        let mut map3 = FxHashMap::default();
        map3.insert(String::from("v0"), 1);
        map3.insert(String::from("v1"), 0);
        let mut map4 = FxHashMap::default();
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
        let result = metadata.ground_parameters_from_yaml(yaml);
        assert!(result.is_ok());
        let (parameters, elements_in_set_variable_array, elements_in_vector_variable_array) =
            result.unwrap();
        let expected_elements_in_set_variable_array =
            vec![vec![(3, 0)], vec![(3, 1)], vec![(3, 0)], vec![(3, 1)]];
        let expected_elements_in_vector_variable_array = vec![vec![], vec![], vec![], vec![]];
        assert_eq!(parameters, expected_parameters);
        assert_eq!(
            elements_in_set_variable_array,
            expected_elements_in_set_variable_array
        );
        assert_eq!(
            elements_in_vector_variable_array,
            expected_elements_in_vector_variable_array
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
        let result = metadata.ground_parameters_from_yaml(yaml);
        assert!(result.is_ok());
        let (parameters, elements_in_set_variable_array, elements_in_vector_variable_array) =
            result.unwrap();
        let expected_elements_in_set_variable_array =
            vec![vec![(3, 0)], vec![(3, 0)], vec![(3, 1)], vec![(3, 1)]];
        let expected_elements_in_vector_variable_array =
            vec![vec![(3, 0)], vec![(3, 1)], vec![(3, 0)], vec![(3, 1)]];
        assert_eq!(parameters, expected_parameters);
        assert_eq!(
            elements_in_set_variable_array,
            expected_elements_in_set_variable_array
        );
        assert_eq!(
            elements_in_vector_variable_array,
            expected_elements_in_vector_variable_array
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
        let result = metadata.ground_parameters_from_yaml(yaml);
        assert!(result.is_err());

        let yaml = yaml_rust::YamlLoader::load_from_str(
            r"
- name: v0
  object: small
- name: v0
  object: s5
",
        );
        assert!(yaml.is_ok());
        let yaml = yaml.unwrap();
        assert_eq!(yaml.len(), 1);
        let yaml = &yaml[0];
        let result = metadata.ground_parameters_from_yaml(yaml);
        assert!(result.is_err());
    }

    #[test]
    fn state_metadata_load_from_yaml_ok() {
        let mut name_to_integer_variable = FxHashMap::default();
        name_to_integer_variable.insert(String::from("n0"), 0);
        let expected = StateMetadata {
            integer_variable_names: vec![String::from("n0")],
            name_to_integer_variable,
            ..Default::default()
        };
        let objects = yaml_rust::Yaml::Array(Vec::new());
        let object_numbers = yaml_rust::Yaml::Hash(linked_hash_map::LinkedHashMap::default());
        let variables = r"
- name: n0
  type: integer
";

        let variables = yaml_rust::YamlLoader::load_from_str(variables);
        assert!(variables.is_ok());
        let variables = variables.unwrap();
        assert_eq!(variables.len(), 1);
        let variables = &variables[0];

        let metadata = StateMetadata::load_from_yaml(&objects, variables, &object_numbers);
        assert!(metadata.is_ok());
        assert_eq!(metadata.unwrap(), expected);

        let expected = generate_metadata();

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
  type: vector
  object: object
- name: p1
  type: vector 
  object: object
- name: p2
  type: vector
  object: object
- name: p3
  type: vector
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

        let metadata = StateMetadata::load_from_yaml(objects, variables, object_numbers);
        assert!(metadata.is_ok());
        assert_eq!(metadata.unwrap(), expected);
    }

    #[test]
    fn state_metadata_load_from_yaml_err() {
        let objects = r"
- object
- object
";
        let variables = r"
- name: s0
  type: set 
  object: object
";
        let object_numbers = r"
object: 10
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

        let metadata = StateMetadata::load_from_yaml(objects, variables, object_numbers);
        assert!(metadata.is_err());

        let objects = r"
- object
- small
";
        let variables = r"
- name: object
  type: set 
  object: object
";
        let object_numbers = r"
object: 10
small: 3
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

        let metadata = StateMetadata::load_from_yaml(objects, variables, object_numbers);
        assert!(metadata.is_err());

        let objects = r"
- object
- small
";
        let variables = r"
- name: s0
  type: set 
  object: object
- name: s0
  type: numeric
";
        let object_numbers = r"
object: 10
small: 3
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

        let metadata = StateMetadata::load_from_yaml(objects, variables, object_numbers);
        assert!(metadata.is_err());

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

        let metadata = StateMetadata::load_from_yaml(objects, variables, object_numbers);
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
        let metadata = StateMetadata::load_from_yaml(objects, variables, object_numbers);
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
        let metadata = StateMetadata::load_from_yaml(objects, variables, object_numbers);
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
        let metadata = StateMetadata::load_from_yaml(objects, variables, object_numbers);
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
        let metadata = StateMetadata::load_from_yaml(objects, variables, object_numbers);
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
        let metadata = StateMetadata::load_from_yaml(objects, variables, object_numbers);
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

        let variables = r"
- name: s0
  type: set
  object: object
- name: n0
  type: integer
  preference: null
";
        let variables = yaml_rust::YamlLoader::load_from_str(variables);
        assert!(variables.is_ok());
        let variables = variables.unwrap();
        assert_eq!(variables.len(), 1);
        let variables = &variables[0];

        let metadata = StateMetadata::load_from_yaml(objects, variables, object_numbers);
        assert!(metadata.is_err());
    }
}
