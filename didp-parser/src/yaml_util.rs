use crate::variable;
use crate::yaml_util;
use std::convert::TryFrom;
use std::error;
use std::fmt;
use std::str;
use yaml_rust::Yaml;

#[derive(Debug, Clone)]
pub struct YamlContentErr(String);

impl YamlContentErr {
    pub fn new(message: String) -> YamlContentErr {
        YamlContentErr(format!("Error in yaml contents: {}", message))
    }
}

impl fmt::Display for YamlContentErr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl error::Error for YamlContentErr {}

pub fn get_map(
    value: &Yaml,
) -> Result<&linked_hash_map::LinkedHashMap<Yaml, Yaml>, YamlContentErr> {
    match value {
        Yaml::Hash(map) => Ok(map),
        _ => Err(YamlContentErr::new(format!(
            "expected Hash, but {:?}",
            value
        ))),
    }
}

pub fn get_map_by_key<'a, 'b>(
    map: &'a linked_hash_map::LinkedHashMap<Yaml, Yaml>,
    key: &'b str,
) -> Result<&'a linked_hash_map::LinkedHashMap<Yaml, Yaml>, YamlContentErr> {
    match map.get(&Yaml::String(key.to_string())) {
        Some(value) => get_map(value),
        None => Err(YamlContentErr::new(format!(
            "no such key `{}` in yaml",
            key
        ))),
    }
}

pub fn get_yaml_by_key<'a, 'b>(
    map: &'a linked_hash_map::LinkedHashMap<Yaml, Yaml>,
    key: &'b str,
) -> Result<&'a Yaml, YamlContentErr> {
    match map.get(&Yaml::String(key.to_string())) {
        Some(value) => Ok(value),
        None => Err(YamlContentErr::new(format!(
            "no such key `{}` in yaml",
            key
        ))),
    }
}

pub fn get_bool(value: &Yaml) -> Result<bool, YamlContentErr> {
    match value {
        Yaml::Boolean(value) => Ok(*value),
        _ => Err(YamlContentErr::new(format!(
            "expected Boolean, but is `{:?}`",
            value
        ))),
    }
}

pub fn get_bool_by_key(
    map: &linked_hash_map::LinkedHashMap<Yaml, Yaml>,
    key: &str,
) -> Result<bool, YamlContentErr> {
    match map.get(&Yaml::String(key.to_string())) {
        Some(value) => get_bool(value),
        None => Err(YamlContentErr::new(format!("key `{}` not found", key))),
    }
}

pub fn get_usize(value: &Yaml) -> Result<usize, YamlContentErr> {
    if let Yaml::Integer(value) = value {
        match variable::ElementVariable::try_from(*value) {
            Ok(value) => Ok(value),
            Err(e) => Err(YamlContentErr::new(format!(
                "cannot convert {} to usize: {:?}",
                value, e
            ))),
        }
    } else {
        Err(YamlContentErr::new(format!(
            "expected Integer, but is `{:?}`",
            value
        )))
    }
}

pub fn get_usize_array(value: &Yaml) -> Result<Vec<usize>, YamlContentErr> {
    if let Yaml::Array(array) = value {
        let mut result = Vec::with_capacity(array.len());
        for value in array {
            result.push(get_usize(value)?);
        }
        Ok(result)
    } else {
        Err(YamlContentErr::new(format!(
            "expected Array, but is `{:?}`",
            value
        )))
    }
}

pub fn get_usize_by_key(
    map: &linked_hash_map::LinkedHashMap<Yaml, Yaml>,
    key: &str,
) -> Result<usize, YamlContentErr> {
    match map.get(&Yaml::String(key.to_string())) {
        Some(value) => get_usize(value),
        None => Err(YamlContentErr::new(format!("key `{}` not found", key))),
    }
}

pub fn get_usize_array_by_key(
    map: &linked_hash_map::LinkedHashMap<Yaml, Yaml>,
    key: &str,
) -> Result<Vec<usize>, YamlContentErr> {
    match map.get(&Yaml::String(key.to_string())) {
        Some(value) => get_usize_array(value),
        None => Err(YamlContentErr::new(format!("key `{}` not found", key))),
    }
}

pub fn get_numeric<T: variable::Numeric>(value: &Yaml) -> Result<T, YamlContentErr>
where
    <T as str::FromStr>::Err: fmt::Debug,
{
    match value {
        Yaml::Integer(value) => match T::from(*value) {
            Some(value) => Ok(value),
            None => Err(yaml_util::YamlContentErr::new(format!(
                "could not parse {} as a number",
                value
            ))),
        },
        Yaml::Real(value) => value.parse().map_err(|e| {
            yaml_util::YamlContentErr::new(format!(
                "could not parse {} as a number: {:?}",
                value, e
            ))
        }),
        _ => Err(yaml_util::YamlContentErr::new(format!(
            "expected Integer or Real, but is {:?}",
            value
        ))),
    }
}

pub fn get_numeric_by_key<T: variable::Numeric>(
    map: &linked_hash_map::LinkedHashMap<Yaml, Yaml>,
    key: &str,
) -> Result<T, YamlContentErr>
where
    <T as str::FromStr>::Err: fmt::Debug,
{
    match map.get(&Yaml::String(key.to_string())) {
        Some(value) => get_numeric(value),
        None => Err(YamlContentErr::new(format!("key `{}` not found", key))),
    }
}

pub fn get_string(value: &Yaml) -> Result<String, YamlContentErr> {
    match value {
        Yaml::String(string) => Ok(string.clone()),
        _ => Err(YamlContentErr::new(format!(
            "expected String, but {:?}",
            value
        ))),
    }
}

pub fn get_string_by_key(
    map: &linked_hash_map::LinkedHashMap<Yaml, Yaml>,
    key: &str,
) -> Result<String, YamlContentErr> {
    match map.get(&Yaml::String(key.to_string())) {
        Some(value) => get_string(value),
        None => Err(YamlContentErr::new(format!("key `{}` not found", key))),
    }
}

pub fn get_string_array_by_key(
    map: &linked_hash_map::LinkedHashMap<Yaml, Yaml>,
    key: &str,
) -> Result<Vec<String>, YamlContentErr> {
    match map.get(&Yaml::String(key.to_string())) {
        Some(Yaml::Array(value)) => parse_string_array(value),
        Some(value) => Err(YamlContentErr::new(format!(
            "the value of key `{}` is not Array, but `{:?}`",
            key, value
        ))),
        None => Err(YamlContentErr::new(format!("key `{}` not found", key))),
    }
}

fn parse_string_array(array: &[Yaml]) -> Result<Vec<String>, YamlContentErr> {
    let mut result = Vec::with_capacity(array.len());
    for v in array {
        result.push(get_string(v)?);
    }
    Ok(result)
}
