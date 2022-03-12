use std::convert::TryFrom;
use std::error;
use std::fmt;
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

pub fn get_hash(
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

pub fn get_hash_value<'a, 'b>(
    map: &'a linked_hash_map::LinkedHashMap<Yaml, Yaml>,
    key: &'b str,
) -> Result<&'a linked_hash_map::LinkedHashMap<Yaml, Yaml>, YamlContentErr> {
    match map.get(&Yaml::String(key.to_string())) {
        Some(value) => get_hash(value),
        None => Err(YamlContentErr::new(format!("key `{}` not found", key))),
    }
}

pub fn get_usize(value: &Yaml) -> Result<usize, Box<dyn error::Error>> {
    match value {
        Yaml::Integer(i) => Ok(usize::try_from(*i)?),
        _ => Err(YamlContentErr::new(format!("expected Integer, but {:?}", value)).into()),
    }
}

pub fn get_string(value: &Yaml) -> Result<String, YamlContentErr> {
    match value {
        Yaml::String(string) => Ok(string.clone()),
        _ => Err(YamlContentErr::new(format!(
            "expected Hash, but {:?}",
            value
        ))),
    }
}

pub fn get_string_value(
    map: &linked_hash_map::LinkedHashMap<Yaml, Yaml>,
    key: &str,
) -> Result<String, YamlContentErr> {
    match map.get(&Yaml::String(key.to_string())) {
        Some(Yaml::String(value)) => Ok(value.clone()),
        Some(value) => Err(YamlContentErr::new(format!(
            "the value of key `{}` is not String, but `{:?}`",
            key, value
        ))),
        None => Err(YamlContentErr::new(format!("key `{}` not found", key))),
    }
}

pub fn get_string_array(
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

pub fn parse_string_array(array: &[Yaml]) -> Result<Vec<String>, YamlContentErr> {
    let mut result = Vec::with_capacity(array.len());
    for v in array {
        match v {
            Yaml::String(string) => result.push(string.clone()),
            _ => {
                return Err(YamlContentErr::new(format!(
                    "expected String but is `{:?}`",
                    v
                )))
            }
        }
    }
    Ok(result)
}
