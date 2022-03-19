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
    match map.get(&Yaml::String(String::from(key))) {
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
    match map.get(&Yaml::String(String::from(key))) {
        Some(value) => Ok(value),
        None => Err(YamlContentErr::new(format!(
            "no such key `{}` in yaml",
            key
        ))),
    }
}

pub fn get_array(value: &Yaml) -> Result<&Vec<Yaml>, YamlContentErr> {
    match value {
        Yaml::Array(array) => Ok(array),
        _ => Err(YamlContentErr::new(format!(
            "expected Array, but is `{:?}`",
            value
        ))),
    }
}

pub fn get_array_by_key<'a, 'b>(
    map: &'a linked_hash_map::LinkedHashMap<Yaml, Yaml>,
    key: &'b str,
) -> Result<&'a Vec<Yaml>, YamlContentErr> {
    match map.get(&Yaml::String(String::from(key))) {
        Some(value) => get_array(value),
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
    match map.get(&Yaml::String(String::from(key))) {
        Some(value) => get_bool(value),
        None => Err(YamlContentErr::new(format!("key `{}` not found", key))),
    }
}

pub fn get_usize(value: &Yaml) -> Result<usize, YamlContentErr> {
    if let Yaml::Integer(value) = value {
        match variable::Element::try_from(*value) {
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
    match map.get(&Yaml::String(String::from(key))) {
        Some(value) => get_usize(value),
        None => Err(YamlContentErr::new(format!("key `{}` not found", key))),
    }
}

pub fn get_usize_array_by_key(
    map: &linked_hash_map::LinkedHashMap<Yaml, Yaml>,
    key: &str,
) -> Result<Vec<usize>, YamlContentErr> {
    match map.get(&Yaml::String(String::from(key))) {
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
    match map.get(&Yaml::String(String::from(key))) {
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

pub fn get_string_array(value: &Yaml) -> Result<Vec<String>, YamlContentErr> {
    match value {
        Yaml::Array(value) => parse_string_array(value),
        _ => Err(YamlContentErr::new(format!(
            "expected Array, but is `{:?}`",
            value
        ))),
    }
}

pub fn get_string_by_key(
    map: &linked_hash_map::LinkedHashMap<Yaml, Yaml>,
    key: &str,
) -> Result<String, YamlContentErr> {
    match map.get(&Yaml::String(String::from(key))) {
        Some(value) => get_string(value),
        None => Err(YamlContentErr::new(format!("key `{}` not found", key))),
    }
}

pub fn get_string_array_by_key(
    map: &linked_hash_map::LinkedHashMap<Yaml, Yaml>,
    key: &str,
) -> Result<Vec<String>, YamlContentErr> {
    match map.get(&Yaml::String(String::from(key))) {
        Some(value) => get_string_array(value),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_map_ok() {
        let mut map = linked_hash_map::LinkedHashMap::<Yaml, Yaml>::new();
        map.insert(Yaml::Integer(0), Yaml::Integer(2));
        let yaml = Yaml::Hash(map);
        let map = get_map(&yaml);
        assert!(map.is_ok());
        let map = map.unwrap();
        assert_eq!(map.len(), 1);
        assert!(matches!(map.get(&Yaml::Integer(0)), Some(Yaml::Integer(2))));
    }

    #[test]
    fn get_map_err() {
        let yaml = Yaml::Array(vec![Yaml::Integer(0), Yaml::Integer(1), Yaml::Integer(2)]);
        let map = get_map(&yaml);
        assert!(map.is_err());
    }

    #[test]
    fn get_map_by_key_ok() {
        let mut inner_map = linked_hash_map::LinkedHashMap::<Yaml, Yaml>::new();
        inner_map.insert(Yaml::Integer(0), Yaml::Integer(2));
        let mut outer_map = linked_hash_map::LinkedHashMap::<Yaml, Yaml>::new();
        outer_map.insert(Yaml::String(String::from("map")), Yaml::Hash(inner_map));
        let map = get_map_by_key(&outer_map, "map");
        assert!(map.is_ok());
        let map = map.unwrap();
        assert_eq!(map.len(), 1);
        assert!(matches!(map.get(&Yaml::Integer(0)), Some(Yaml::Integer(2))));
    }

    #[test]
    fn get_map_by_key_err() {
        let mut inner_map = linked_hash_map::LinkedHashMap::<Yaml, Yaml>::new();
        inner_map.insert(Yaml::Integer(0), Yaml::Integer(2));
        let mut outer_map = linked_hash_map::LinkedHashMap::<Yaml, Yaml>::new();
        outer_map.insert(Yaml::String(String::from("map")), Yaml::Hash(inner_map));
        outer_map.insert(Yaml::String(String::from("integer")), Yaml::Integer(0));
        let map = get_map_by_key(&outer_map, "array");
        assert!(map.is_err());
        let map = get_map_by_key(&outer_map, "integer");
        assert!(map.is_err());
    }

    #[test]
    fn get_yaml_by_key_ok() {
        let mut map = linked_hash_map::LinkedHashMap::<Yaml, Yaml>::new();
        map.insert(Yaml::String(String::from("yaml")), Yaml::Integer(2));
        let yaml = get_yaml_by_key(&map, "yaml");
        assert!(yaml.is_ok());
        assert!(matches!(yaml.unwrap(), Yaml::Integer(2)));
    }

    #[test]
    fn get_yaml_by_key_err() {
        let mut map = linked_hash_map::LinkedHashMap::<Yaml, Yaml>::new();
        map.insert(Yaml::String(String::from("yaml")), Yaml::Integer(2));
        let yaml = get_yaml_by_key(&map, "json");
        assert!(yaml.is_err());
    }

    #[test]
    fn get_array_ok() {
        let yaml = Yaml::Array(vec![Yaml::Integer(0), Yaml::Integer(1)]);
        let array = get_array(&yaml);
        assert!(array.is_ok());
        let array = array.unwrap();
        assert_eq!(array.len(), 2);
        assert!(matches!(array[0], Yaml::Integer(0)));
        assert!(matches!(array[1], Yaml::Integer(1)));
    }

    #[test]
    fn get_array_err() {
        let yaml = Yaml::Integer(0);
        let array = get_array(&yaml);
        assert!(array.is_err());
    }

    #[test]
    fn get_array_by_key_ok() {
        let mut map = linked_hash_map::LinkedHashMap::<Yaml, Yaml>::new();
        map.insert(
            Yaml::String(String::from("array")),
            Yaml::Array(vec![Yaml::Integer(0), Yaml::Integer(1)]),
        );
        let array = get_array_by_key(&map, "array");
        assert!(array.is_ok());
        let array = array.unwrap();
        assert_eq!(array.len(), 2);
        assert!(matches!(array[0], Yaml::Integer(0)));
        assert!(matches!(array[1], Yaml::Integer(1)));
    }

    #[test]
    fn get_array_by_key_err() {
        let mut map = linked_hash_map::LinkedHashMap::<Yaml, Yaml>::new();
        map.insert(
            Yaml::String(String::from("array")),
            Yaml::Array(vec![Yaml::Integer(0), Yaml::Integer(1)]),
        );
        map.insert(Yaml::String(String::from("integer")), Yaml::Integer(0));
        let array = get_array_by_key(&map, "map");
        assert!(array.is_err());
        let array = get_array_by_key(&map, "integer");
        assert!(array.is_err());
    }

    #[test]
    fn get_bool_ok() {
        let yaml = Yaml::Boolean(true);
        let result = get_bool(&yaml);
        assert!(result.is_ok());
        assert!(result.unwrap());
    }

    #[test]
    fn get_bool_err() {
        let yaml = Yaml::Integer(0);
        let result = get_bool(&yaml);
        assert!(result.is_err());
    }

    #[test]
    fn get_bool_by_key_ok() {
        let mut map = linked_hash_map::LinkedHashMap::<Yaml, Yaml>::new();
        map.insert(Yaml::String(String::from("bool")), Yaml::Boolean(true));
        let result = get_bool_by_key(&map, "bool");
        assert!(result.is_ok());
        assert!(result.unwrap());
    }

    #[test]
    fn get_bool_by_key_err() {
        let mut map = linked_hash_map::LinkedHashMap::<Yaml, Yaml>::new();
        map.insert(Yaml::String(String::from("integer")), Yaml::Integer(0));
        let result = get_bool_by_key(&map, "array");
        assert!(result.is_err());
        let result = get_bool_by_key(&map, "integer");
        assert!(result.is_err());
    }

    #[test]
    fn get_usize_ok() {
        let yaml = Yaml::Integer(0);
        let result = get_usize(&yaml);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0);
    }

    #[test]
    fn get_usize_err() {
        let yaml = Yaml::Real(String::from("0.0"));
        let result = get_usize(&yaml);
        assert!(result.is_err());

        let yaml = Yaml::Integer(-1);
        let result = get_usize(&yaml);
        assert!(result.is_err());
    }

    #[test]
    fn get_usize_array_ok() {
        let yaml = Yaml::Array(vec![Yaml::Integer(0), Yaml::Integer(1)]);
        let array = get_usize_array(&yaml);
        assert!(array.is_ok());
        assert_eq!(array.unwrap(), vec![0, 1]);
    }

    #[test]
    fn get_usize_array_err() {
        let yaml = Yaml::Integer(0);
        let array = get_usize_array(&yaml);
        assert!(array.is_err());

        let yaml = Yaml::Array(vec![Yaml::Integer(0), Yaml::Boolean(true)]);
        let array = get_usize_array(&yaml);
        assert!(array.is_err());

        let yaml = Yaml::Array(vec![Yaml::Integer(0), Yaml::Integer(-1)]);
        let array = get_usize_array(&yaml);
        assert!(array.is_err());
    }

    #[test]
    fn get_usize_by_key_ok() {
        let mut map = linked_hash_map::LinkedHashMap::<Yaml, Yaml>::new();
        map.insert(Yaml::String(String::from("usize")), Yaml::Integer(0));
        let result = get_usize_by_key(&map, "usize");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0);
    }

    #[test]
    fn get_usize_by_key_err() {
        let mut map = linked_hash_map::LinkedHashMap::<Yaml, Yaml>::new();
        map.insert(Yaml::String(String::from("bool")), Yaml::Boolean(true));
        map.insert(Yaml::String(String::from("integer")), Yaml::Integer(-1));
        let result = get_usize_by_key(&map, "array");
        assert!(result.is_err());
        let result = get_usize_by_key(&map, "bool");
        assert!(result.is_err());
        let result = get_usize_by_key(&map, "integer");
        assert!(result.is_err());
    }

    #[test]
    fn get_usize_array_by_key_ok() {
        let mut map = linked_hash_map::LinkedHashMap::<Yaml, Yaml>::new();
        map.insert(
            Yaml::String(String::from("array")),
            Yaml::Array(vec![Yaml::Integer(0), Yaml::Integer(1)]),
        );
        let array = get_usize_array_by_key(&map, "array");
        assert!(array.is_ok());
        assert_eq!(array.unwrap(), vec![0, 1]);
    }

    #[test]
    fn get_usize_array_by_key_err() {
        let mut map = linked_hash_map::LinkedHashMap::<Yaml, Yaml>::new();
        map.insert(
            Yaml::String(String::from("array1")),
            Yaml::Array(vec![Yaml::Integer(0), Yaml::Boolean(true)]),
        );
        map.insert(
            Yaml::String(String::from("array2")),
            Yaml::Array(vec![Yaml::Integer(0), Yaml::Integer(-1)]),
        );
        let array = get_usize_array_by_key(&map, "array");
        assert!(array.is_err());
        let array = get_usize_array_by_key(&map, "array1");
        assert!(array.is_err());
        let array = get_usize_array_by_key(&map, "array2");
        assert!(array.is_err());
    }

    #[test]
    fn get_numeric_ok() {
        let yaml = Yaml::Integer(0);
        let result = get_numeric::<variable::Integer>(&yaml);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0);
    }

    #[test]
    fn get_numeric_err() {
        let yaml = Yaml::Real(String::from("0.5"));
        let result = get_numeric::<variable::Integer>(&yaml);
        assert!(result.is_err());
        let yaml = Yaml::Boolean(true);
        let result = get_numeric::<variable::Integer>(&yaml);
        assert!(result.is_err());
    }

    #[test]
    fn get_numeric_by_key_ok() {
        let mut map = linked_hash_map::LinkedHashMap::<Yaml, Yaml>::new();
        map.insert(Yaml::String(String::from("numeric")), Yaml::Integer(0));
        let result = get_numeric_by_key::<variable::Integer>(&map, "numeric");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0);
    }

    #[test]
    fn get_numeric_by_key_err() {
        let mut map = linked_hash_map::LinkedHashMap::<Yaml, Yaml>::new();
        map.insert(
            Yaml::String(String::from("numeric")),
            Yaml::Real(String::from("0.5")),
        );
        map.insert(Yaml::String(String::from("bool")), Yaml::Boolean(true));
        let result = get_numeric_by_key::<variable::Integer>(&map, "integer");
        assert!(result.is_err());
        let result = get_numeric_by_key::<variable::Integer>(&map, "bool");
        assert!(result.is_err());
        let result = get_numeric_by_key::<variable::Integer>(&map, "numeric");
        assert!(result.is_err());
    }

    #[test]
    fn get_string_ok() {
        let yaml = Yaml::String(String::from("string"));
        let result = get_string(&yaml);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), String::from("string"));
    }

    #[test]
    fn get_string_err() {
        let yaml = Yaml::Real(String::from("0.0"));
        let result = get_string(&yaml);
        assert!(result.is_err());
    }

    #[test]
    fn get_string_array_ok() {
        let yaml = Yaml::Array(vec![
            Yaml::String(String::from("0")),
            Yaml::String(String::from("1")),
        ]);
        let array = get_string_array(&yaml);
        assert!(array.is_ok());
        assert_eq!(array.unwrap(), vec![String::from("0"), String::from("1")]);
    }

    #[test]
    fn get_string_array_err() {
        let yaml = Yaml::Integer(0);
        let array = get_string_array(&yaml);
        assert!(array.is_err());

        let yaml = Yaml::Array(vec![Yaml::String(String::from("0")), Yaml::Boolean(true)]);
        let array = get_string_array(&yaml);
        assert!(array.is_err());
    }

    #[test]
    fn get_string_by_key_ok() {
        let mut map = linked_hash_map::LinkedHashMap::<Yaml, Yaml>::new();
        map.insert(
            Yaml::String(String::from("string")),
            Yaml::String(String::from("0")),
        );
        let result = get_string_by_key(&map, "string");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), String::from("0"));
    }

    #[test]
    fn get_string_by_key_err() {
        let mut map = linked_hash_map::LinkedHashMap::<Yaml, Yaml>::new();
        map.insert(Yaml::String(String::from("bool")), Yaml::Boolean(true));
        let result = get_string_by_key(&map, "array");
        assert!(result.is_err());
        let result = get_string_by_key(&map, "bool");
        assert!(result.is_err());
    }

    #[test]
    fn get_string_array_by_key_ok() {
        let mut map = linked_hash_map::LinkedHashMap::<Yaml, Yaml>::new();
        map.insert(
            Yaml::String(String::from("array")),
            Yaml::Array(vec![
                Yaml::String(String::from("0")),
                Yaml::String(String::from("1")),
            ]),
        );
        let array = get_string_array_by_key(&map, "array");
        assert!(array.is_ok());
        assert_eq!(array.unwrap(), vec![String::from("0"), String::from("1")]);
    }

    #[test]
    fn get_string_array_by_key_err() {
        let mut map = linked_hash_map::LinkedHashMap::<Yaml, Yaml>::new();
        map.insert(
            Yaml::String(String::from("array")),
            Yaml::Array(vec![Yaml::String(String::from("0")), Yaml::Boolean(true)]),
        );
        let array = get_string_array_by_key(&map, "map");
        assert!(array.is_err());
        let array = get_string_array_by_key(&map, "array");
        assert!(array.is_err());
    }
}
