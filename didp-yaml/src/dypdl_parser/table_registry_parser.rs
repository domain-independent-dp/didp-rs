use crate::util;
use dypdl::prelude::*;
use dypdl::{StateMetadata, TableRegistry};
use lazy_static::lazy_static;
use rustc_hash::FxHashMap;
use std::error::Error;
use std::fmt;
use std::str;
use yaml_rust::Yaml;

enum TableReturnType {
    Integer(Integer),
    Continuous(Continuous),
    Set(Set),
    Vector(usize, Vector),
    Element(Element),
    Bool(bool),
}

/// Returns tables of constants loaded from YAML
///
/// # Errors
///
/// If the format is invalid.
pub fn load_table_registry_from_yaml(
    tables: &Yaml,
    table_values: &Yaml,
    dictionaries: &Yaml,
    dictionary_values: &Yaml,
    metadata: &StateMetadata,
) -> Result<TableRegistry, Box<dyn Error>> {
    lazy_static! {
        static ref ARGS_KEY: yaml_rust::Yaml = yaml_rust::Yaml::from_str("args");
    }
    // Get the information of all tables in the domain file.
    let tables = util::get_array(tables)?;
    let mut table_names = Vec::with_capacity(tables.len());
    let mut name_to_signature = FxHashMap::default();
    let mut reserved_names = metadata.get_name_set();
    for value in tables {
        let map = util::get_map(value)?;
        let name = util::get_string_by_key(map, "name")?;
        if let Some(name) = reserved_names.get(&name) {
            return Err(util::YamlContentErr::new(format!(
                "table name `{}` is already used",
                name
            ))
            .into());
        }
        reserved_names.insert(name.clone());
        let dimension_sizes = match map.get(&ARGS_KEY) {
            Some(Yaml::Array(arg_array)) => util::get_table_arg_array(arg_array, metadata),
            Some(value) => Err(util::YamlContentErr::new(format!(
                "expected Array, but is `{:?}`",
                value
            ))
            .into()),
            None => Ok(Vec::new()),
        }?;

        let return_type = util::get_string_by_key(map, "type")?;
        match &return_type[..] {
            "integer" => {
                if let Ok(value) = util::get_numeric_by_key(map, "default") {
                    name_to_signature.insert(
                        name.clone(),
                        (dimension_sizes, TableReturnType::Integer(value)),
                    );
                } else {
                    name_to_signature
                        .insert(name.clone(), (dimension_sizes, TableReturnType::Integer(0)));
                }
            }
            "continuous" => {
                if let Ok(value) = util::get_numeric_by_key(map, "default") {
                    name_to_signature.insert(
                        name.clone(),
                        (dimension_sizes, TableReturnType::Continuous(value)),
                    );
                } else {
                    name_to_signature.insert(
                        name.clone(),
                        (dimension_sizes, TableReturnType::Continuous(0.0)),
                    );
                }
            }
            "set" => {
                let object = match map.get(&Yaml::from_str("object")) {
                    Some(value) => Ok(value),
                    None => Err(util::YamlContentErr::new(format!(
                        "object not found for set table {}",
                        name
                    ))),
                }?;
                let n = util::get_size_from_yaml(object, metadata)?;
                let mut default = Set::with_capacity(n);
                if let Ok(array) = util::get_usize_array_by_key(map, "default") {
                    for v in array {
                        if v >= n {
                            return Err(util::YamlContentErr::new(format!(
                                "element `{}` is too large for the set table `{}`",
                                v, name
                            ))
                            .into());
                        }
                        default.insert(v);
                    }
                }
                name_to_signature.insert(
                    name.clone(),
                    (dimension_sizes, TableReturnType::Set(default)),
                );
            }
            "vector" => {
                let object_name = util::get_string_by_key(map, "object")?;
                let object = match metadata.name_to_object_type.get(&object_name) {
                    Some(object) => *object,
                    None => {
                        return Err(util::YamlContentErr::new(format!(
                            "no such object `{}`",
                            object_name
                        ))
                        .into())
                    }
                };
                let n = metadata.object_numbers[object];
                let default = match util::get_usize_array_by_key(map, "default") {
                    Ok(array) => {
                        for v in &array {
                            if *v >= n {
                                return Err(util::YamlContentErr::new(format!(
                                    "element `{}` is too large for object `{}`",
                                    *v, object_name
                                ))
                                .into());
                            }
                        }
                        array
                    }
                    _ => Vec::new(),
                };
                name_to_signature.insert(
                    name.clone(),
                    (dimension_sizes, TableReturnType::Vector(n, default)),
                );
            }
            "element" => {
                if let Ok(value) = util::get_usize_by_key(map, "default") {
                    name_to_signature.insert(
                        name.clone(),
                        (dimension_sizes, TableReturnType::Element(value)),
                    );
                } else {
                    name_to_signature
                        .insert(name.clone(), (dimension_sizes, TableReturnType::Element(0)));
                }
            }
            "bool" => {
                if let Ok(value) = util::get_bool_by_key(map, "default") {
                    name_to_signature.insert(
                        name.clone(),
                        (dimension_sizes, TableReturnType::Bool(value)),
                    );
                } else {
                    name_to_signature.insert(
                        name.clone(),
                        (dimension_sizes, TableReturnType::Bool(false)),
                    );
                }
            }
            _ => {
                return Err(util::YamlContentErr::new(format!(
                    "no such table type `{}`",
                    return_type
                ))
                .into())
            }
        }
        table_names.push(name);
    }

    // Insert all tables and their values into the table registry.
    let mut registry = TableRegistry::default();
    let table_values = util::get_map(table_values)?;
    for name in table_names {
        let (dimension_sizes, return_type) = name_to_signature.get(&name).unwrap();
        let value = util::get_yaml_by_key(table_values, &name)?;
        if dimension_sizes.is_empty() {
            match return_type {
                TableReturnType::Integer(_) => {
                    registry
                        .integer_tables
                        .name_to_constant
                        .insert(name, util::get_numeric(value)?);
                }
                TableReturnType::Continuous(_) => {
                    registry
                        .continuous_tables
                        .name_to_constant
                        .insert(name, util::get_numeric(value)?);
                }
                TableReturnType::Set(default) => {
                    let value = load_set_from_yaml(value, default.len())?;
                    registry.set_tables.name_to_constant.insert(name, value);
                }
                TableReturnType::Vector(capacity, _) => {
                    let value = load_vector_from_yaml(value, *capacity)?;
                    registry.vector_tables.name_to_constant.insert(name, value);
                }
                TableReturnType::Element(_) => {
                    registry
                        .element_tables
                        .name_to_constant
                        .insert(name, util::get_usize(value)?);
                }
                TableReturnType::Bool(_) => {
                    registry
                        .bool_tables
                        .name_to_constant
                        .insert(name, util::get_bool(value)?);
                }
            }
        } else if dimension_sizes.len() == 1 {
            let size = dimension_sizes[0];
            match return_type {
                TableReturnType::Integer(default) => {
                    let f = load_numeric_table_1d_from_yaml(value, size, *default)?;
                    registry.add_table_1d(name, f)?;
                }
                TableReturnType::Continuous(default) => {
                    let f = load_numeric_table_1d_from_yaml(value, size, *default)?;
                    registry.add_table_1d(name, f)?;
                }
                TableReturnType::Set(default) => {
                    let f = load_set_table_1d_from_yaml(value, size, default)?;
                    registry.add_table_1d(name, f)?;
                }
                TableReturnType::Vector(capacity, default) => {
                    let f = load_vector_table_1d_from_yaml(value, size, default, *capacity)?;
                    registry.add_table_1d(name, f)?;
                }
                TableReturnType::Element(default) => {
                    let f = load_numeric_table_1d_from_yaml(value, size, *default)?;
                    registry.add_table_1d(name, f)?;
                }
                TableReturnType::Bool(default) => {
                    let f = load_bool_table_1d_from_yaml(value, size, *default)?;
                    registry.add_table_1d(name, f)?;
                }
            }
        } else if dimension_sizes.len() == 2 {
            let size_x = dimension_sizes[0];
            let size_y = dimension_sizes[1];
            match return_type {
                TableReturnType::Integer(default) => {
                    let f = load_numeric_table_2d_from_yaml(value, size_x, size_y, *default)?;
                    registry.add_table_2d(name, f)?;
                }
                TableReturnType::Continuous(default) => {
                    let f = load_numeric_table_2d_from_yaml(value, size_x, size_y, *default)?;
                    registry.add_table_2d(name, f)?;
                }
                TableReturnType::Set(default) => {
                    let f = load_set_table_2d_from_yaml(value, size_x, size_y, default)?;
                    registry.add_table_2d(name, f)?;
                }
                TableReturnType::Vector(capacity, default) => {
                    let f =
                        load_vector_table_2d_from_yaml(value, size_x, size_y, default, *capacity)?;
                    registry.add_table_2d(name, f)?;
                }
                TableReturnType::Element(default) => {
                    let f = load_numeric_table_2d_from_yaml(value, size_x, size_y, *default)?;
                    registry.add_table_2d(name, f)?;
                }
                TableReturnType::Bool(default) => {
                    let f = load_bool_table_2d_from_yaml(value, size_x, size_y, *default)?;
                    registry.add_table_2d(name, f)?;
                }
            }
        } else if dimension_sizes.len() == 3 {
            let size_x = dimension_sizes[0];
            let size_y = dimension_sizes[1];
            let size_z = dimension_sizes[2];
            match return_type {
                TableReturnType::Integer(default) => {
                    let f =
                        load_numeric_table_3d_from_yaml(value, size_x, size_y, size_z, *default)?;
                    registry.add_table_3d(name, f)?;
                }
                TableReturnType::Continuous(default) => {
                    let f =
                        load_numeric_table_3d_from_yaml(value, size_x, size_y, size_z, *default)?;
                    registry.add_table_3d(name, f)?;
                }
                TableReturnType::Set(default) => {
                    let f = load_set_table_3d_from_yaml(value, size_x, size_y, size_z, default)?;
                    registry.add_table_3d(name, f)?;
                }
                TableReturnType::Vector(capacity, default) => {
                    let f = load_vector_table_3d_from_yaml(
                        value, size_x, size_y, size_z, default, *capacity,
                    )?;
                    registry.add_table_3d(name, f)?;
                }
                TableReturnType::Element(default) => {
                    let f =
                        load_numeric_table_3d_from_yaml(value, size_x, size_y, size_z, *default)?;
                    registry.add_table_3d(name, f)?;
                }
                TableReturnType::Bool(default) => {
                    let f = load_bool_table_3d_from_yaml(value, size_x, size_y, size_z, *default)?;
                    registry.add_table_3d(name, f)?;
                }
            }
        } else {
            let size = dimension_sizes.clone();
            match return_type {
                TableReturnType::Integer(default) => {
                    let (f, default) = load_numeric_table_from_yaml(value, size, *default)?;
                    registry.add_table(name, f, default)?;
                }
                TableReturnType::Continuous(default) => {
                    let (f, default) = load_numeric_table_from_yaml(value, size, *default)?;
                    registry.add_table(name, f, default)?;
                }
                TableReturnType::Set(default) => {
                    let (f, default) = load_set_table_from_yaml(value, size, default.clone())?;
                    registry.add_table(name, f, default)?;
                }
                TableReturnType::Vector(capacity, default) => {
                    let (f, default) =
                        load_vector_table_from_yaml(value, size, default.clone(), *capacity)?;
                    registry.add_table(name, f, default)?;
                }
                TableReturnType::Element(default) => {
                    let (f, default) = load_numeric_table_from_yaml(value, size, *default)?;
                    registry.add_table(name, f, default)?;
                }
                TableReturnType::Bool(default) => {
                    let (f, default) = load_bool_table_from_yaml(value, size, *default)?;
                    registry.add_table(name, f, default)?;
                }
            }
        }
    }

    // Get the information of all dictionaries in the domain file.
    let dictionaries = util::get_array(dictionaries)?;
    let mut dictionary_names = Vec::with_capacity(dictionaries.len());
    let mut name_to_signature = FxHashMap::default();
    for value in dictionaries {
        let map = util::get_map(value)?;
        let name = util::get_string_by_key(map, "name")?;
        if let Some(name) = reserved_names.get(&name) {
            return Err(util::YamlContentErr::new(format!(
                "dictionary name `{}` is already used",
                name
            ))
            .into());
        }
        reserved_names.insert(name.clone());

        let return_type = util::get_string_by_key(map, "type")?;
        match &return_type[..] {
            "integer" => {
                if let Ok(value) = util::get_numeric_by_key(map, "default") {
                    name_to_signature.insert(name.clone(), TableReturnType::Integer(value));
                } else {
                    name_to_signature.insert(name.clone(), TableReturnType::Integer(0));
                }
            }
            "continuous" => {
                if let Ok(value) = util::get_numeric_by_key(map, "default") {
                    name_to_signature.insert(name.clone(), TableReturnType::Continuous(value));
                } else {
                    name_to_signature.insert(name.clone(), TableReturnType::Continuous(0.0));
                }
            }
            "set" => {
                let object = match map.get(&Yaml::from_str("object")) {
                    Some(value) => Ok(value),
                    None => Err(util::YamlContentErr::new(format!(
                        "object not found for set dictionary {}",
                        name
                    ))),
                }?;
                let n = util::get_size_from_yaml(object, metadata)?;
                let mut default = Set::with_capacity(n);
                if let Ok(array) = util::get_usize_array_by_key(map, "default") {
                    for v in array {
                        if v >= n {
                            return Err(util::YamlContentErr::new(format!(
                                "element `{}` is too large for the set dictionary `{}`",
                                v, name
                            ))
                            .into());
                        }
                        default.insert(v);
                    }
                }
                name_to_signature.insert(name.clone(), TableReturnType::Set(default));
            }
            "vector" => {
                let object_name = util::get_string_by_key(map, "object")?;
                let object = match metadata.name_to_object_type.get(&object_name) {
                    Some(object) => *object,
                    None => {
                        return Err(util::YamlContentErr::new(format!(
                            "no such object `{}`",
                            object_name
                        ))
                        .into())
                    }
                };
                let n = metadata.object_numbers[object];
                let default = match util::get_usize_array_by_key(map, "default") {
                    Ok(array) => {
                        for v in &array {
                            if *v >= n {
                                return Err(util::YamlContentErr::new(format!(
                                    "element `{}` is too large for object `{}`",
                                    *v, object_name
                                ))
                                .into());
                            }
                        }
                        array
                    }
                    _ => Vec::new(),
                };
                name_to_signature.insert(name.clone(), TableReturnType::Vector(n, default));
            }
            "element" => {
                if let Ok(value) = util::get_usize_by_key(map, "default") {
                    name_to_signature.insert(name.clone(), TableReturnType::Element(value));
                } else {
                    name_to_signature.insert(name.clone(), TableReturnType::Element(0));
                }
            }
            "bool" => {
                if let Ok(value) = util::get_bool_by_key(map, "default") {
                    name_to_signature.insert(name.clone(), TableReturnType::Bool(value));
                } else {
                    name_to_signature.insert(name.clone(), TableReturnType::Bool(false));
                }
            }
            _ => {
                return Err(util::YamlContentErr::new(format!(
                    "no such dictionary type `{}`",
                    return_type
                ))
                .into())
            }
        }
        dictionary_names.push(name);
    }

    // Insert all tables and their values into the table registry.
    let dictionary_values = util::get_map(dictionary_values)?;
    for name in dictionary_names {
        let return_type = name_to_signature.get(&name).unwrap();
        let value = util::get_yaml_by_key(dictionary_values, &name)?;

        match return_type {
            TableReturnType::Integer(default) => {
                let (f, default) = load_numeric_dictionary_from_yaml(value, *default)?;
                registry.add_table(name, f, default)?;
            }
            TableReturnType::Continuous(default) => {
                let (f, default) = load_numeric_dictionary_from_yaml(value, *default)?;
                registry.add_table(name, f, default)?;
            }
            TableReturnType::Set(default) => {
                let (f, default) = load_set_dictionary_from_yaml(value, default.clone())?;
                registry.add_table(name, f, default)?;
            }
            TableReturnType::Vector(capacity, default) => {
                let (f, default) =
                    load_vector_dictionary_from_yaml(value, default.clone(), *capacity)?;
                registry.add_table(name, f, default)?;
            }
            TableReturnType::Element(default) => {
                let (f, default) = load_numeric_dictionary_from_yaml(value, *default)?;
                registry.add_table(name, f, default)?;
            }
            TableReturnType::Bool(default) => {
                let (f, default) = load_bool_dictionary_from_yaml(value, *default)?;
                registry.add_table(name, f, default)?;
            }
        }
    }

    Ok(registry)
}

fn load_numeric_table_1d_from_yaml<T: str::FromStr + num_traits::FromPrimitive + Copy>(
    value: &Yaml,
    size: usize,
    default: T,
) -> Result<Vec<T>, util::YamlContentErr>
where
    <T as str::FromStr>::Err: fmt::Debug,
{
    let mut body: Vec<T> = (0..size).map(|_| default).collect();
    let map = util::get_map(value)?;
    for (args, value) in map {
        let args = util::get_usize(args)?;
        let value = util::get_numeric(value)?;
        if args >= size {
            return Err(util::YamlContentErr::new(format!(
                "`{}` is greater than the number of the objects for table",
                args,
            )));
        }
        body[args] = value;
    }
    Ok(body)
}

fn load_numeric_table_2d_from_yaml<T: str::FromStr + num_traits::FromPrimitive + Copy>(
    value: &Yaml,
    size_x: usize,
    size_y: usize,
    default: T,
) -> Result<Vec<Vec<T>>, util::YamlContentErr>
where
    <T as str::FromStr>::Err: fmt::Debug,
{
    let mut body: Vec<Vec<T>> = (0..size_x)
        .map(|_| (0..size_y).map(|_| default).collect())
        .collect();
    let map = util::get_map(value)?;
    for (args, value) in map {
        let args = util::get_usize_array(args)?;
        let x = args[0];
        let y = args[1];
        let value = util::get_numeric(value)?;
        if x >= size_x || y >= size_y {
            return Err(util::YamlContentErr::new(format!(
                "`({}, {})` is greater than the numbers of objects for table",
                x, y,
            )));
        }
        body[x][y] = value;
    }
    Ok(body)
}

fn load_numeric_table_3d_from_yaml<T: str::FromStr + num_traits::FromPrimitive + Copy>(
    value: &Yaml,
    size_x: usize,
    size_y: usize,
    size_z: usize,
    default: T,
) -> Result<Vec<Vec<Vec<T>>>, util::YamlContentErr>
where
    <T as str::FromStr>::Err: fmt::Debug,
{
    let mut body: Vec<Vec<Vec<T>>> = (0..size_x)
        .map(|_| {
            (0..size_y)
                .map(|_| (0..size_z).map(|_| default).collect())
                .collect()
        })
        .collect();
    let map = util::get_map(value)?;
    for (args, value) in map {
        let args = util::get_usize_array(args)?;
        let x = args[0];
        let y = args[1];
        let z = args[2];
        let value = util::get_numeric(value)?;
        if x >= size_x || y >= size_y || z >= size_z {
            return Err(util::YamlContentErr::new(format!(
                "`({}, {}, {})` is greater than the numbers of objects for table",
                x, y, z,
            )));
        }
        body[x][y][z] = value;
    }
    Ok(body)
}

fn load_numeric_table_from_yaml<T: str::FromStr + num_traits::FromPrimitive>(
    value: &Yaml,
    size: Vec<usize>,
    default: T,
) -> Result<(FxHashMap<Vec<Element>, T>, T), util::YamlContentErr>
where
    <T as str::FromStr>::Err: fmt::Debug,
{
    let map = util::get_map(value)?;
    let mut body = FxHashMap::default();
    for (args, value) in map {
        let args = util::get_usize_array(args)?;
        if args.len() != size.len() {
            return Err(util::YamlContentErr::new(format!(
                "expected `{}` arguments for table, but passed `{}`",
                size.len(),
                args.len()
            )));
        }
        let value = util::get_numeric(value)?;
        if args.iter().zip(size.iter()).any(|(a, b)| a >= b) {
            return Err(util::YamlContentErr::new(format!(
                "`{:?}` is greater than the numbers of objects for table",
                args,
            )));
        }
        body.insert(args, value);
    }
    Ok((body, default))
}

fn load_numeric_dictionary_from_yaml<T: str::FromStr + num_traits::FromPrimitive>(
    value: &Yaml,
    default: T,
) -> Result<(FxHashMap<Vec<Element>, T>, T), util::YamlContentErr>
where
    <T as str::FromStr>::Err: fmt::Debug,
{
    let map = util::get_map(value)?;
    let mut body = FxHashMap::default();

    for (args, value) in map {
        let args = util::get_usize_array(args)?;
        let value = util::get_numeric(value)?;
        body.insert(args, value);
    }
    Ok((body, default))
}

fn load_bool_table_1d_from_yaml(
    value: &Yaml,
    size: usize,
    default: bool,
) -> Result<Vec<bool>, util::YamlContentErr> {
    let mut body: Vec<bool> = (0..size).map(|_| default).collect();
    let map = util::get_map(value)?;
    for (args, value) in map {
        let args = util::get_usize(args)?;
        let value = util::get_bool(value)?;
        if args >= size {
            return Err(util::YamlContentErr::new(format!(
                "`{}` is greater than the number of the objects for table",
                args,
            )));
        }
        body[args] = value;
    }
    Ok(body)
}

fn load_bool_table_2d_from_yaml(
    value: &Yaml,
    size_x: usize,
    size_y: usize,
    default: bool,
) -> Result<Vec<Vec<bool>>, util::YamlContentErr> {
    let mut body: Vec<Vec<bool>> = (0..size_x)
        .map(|_| (0..size_y).map(|_| default).collect())
        .collect();
    let map = util::get_map(value)?;
    for (args, value) in map {
        let args = util::get_usize_array(args)?;
        let x = args[0];
        let y = args[1];
        let value = util::get_bool(value)?;
        if x >= size_x || y >= size_y {
            return Err(util::YamlContentErr::new(format!(
                "`({}, {})` is greater than the numbers of objects for table",
                x, y,
            )));
        }
        body[x][y] = value;
    }
    Ok(body)
}

fn load_bool_table_3d_from_yaml(
    value: &Yaml,
    size_x: usize,
    size_y: usize,
    size_z: usize,
    default: bool,
) -> Result<Vec<Vec<Vec<bool>>>, util::YamlContentErr> {
    let mut body: Vec<Vec<Vec<bool>>> = (0..size_x)
        .map(|_| {
            (0..size_y)
                .map(|_| (0..size_z).map(|_| default).collect())
                .collect()
        })
        .collect();
    let map = util::get_map(value)?;
    for (args, value) in map {
        let args = util::get_usize_array(args)?;
        let x = args[0];
        let y = args[1];
        let z = args[2];
        let value = util::get_bool(value)?;
        if x >= size_x || y >= size_y || z >= size_z {
            return Err(util::YamlContentErr::new(format!(
                "`({}, {}, {})` is greater than the numbers of objects for table",
                x, y, z,
            )));
        }
        body[x][y][z] = value;
    }
    Ok(body)
}

fn load_bool_table_from_yaml(
    value: &Yaml,
    size: Vec<usize>,
    default: bool,
) -> Result<(FxHashMap<Vec<Element>, bool>, bool), util::YamlContentErr> {
    let map = util::get_map(value)?;
    let mut body = FxHashMap::default();
    for (args, value) in map {
        let args = util::get_usize_array(args)?;
        if args.len() != size.len() {
            return Err(util::YamlContentErr::new(format!(
                "expected `{}` arguments for table, but passed `{}`",
                size.len(),
                args.len()
            )));
        }
        let value = util::get_bool(value)?;
        if args.iter().zip(size.iter()).any(|(a, b)| a >= b) {
            return Err(util::YamlContentErr::new(format!(
                "`{:?}` is greater than the numbers of objects for table",
                args,
            )));
        }
        body.insert(args, value);
    }
    Ok((body, default))
}

fn load_bool_dictionary_from_yaml(
    value: &Yaml,
    default: bool,
) -> Result<(FxHashMap<Vec<Element>, bool>, bool), util::YamlContentErr> {
    let map = util::get_map(value)?;
    let mut body = FxHashMap::default();

    for (args, value) in map {
        let args = util::get_usize_array(args)?;
        let value = util::get_bool(value)?;
        body.insert(args, value);
    }
    Ok((body, default))
}

fn load_set_from_yaml(value: &Yaml, capacity: usize) -> Result<Set, util::YamlContentErr> {
    let array = util::get_usize_array(value)?;
    let mut set = Set::with_capacity(capacity);
    for v in array {
        if v >= capacity {
            return Err(util::YamlContentErr::new(format!(
                "element `{}` in a set table is too large for the object",
                v,
            )));
        }
        set.insert(v);
    }
    Ok(set)
}

fn load_set_table_1d_from_yaml(
    value: &Yaml,
    size: usize,
    default: &Set,
) -> Result<Vec<Set>, util::YamlContentErr> {
    let mut body: Vec<Set> = (0..size).map(|_| default.clone()).collect();
    let map = util::get_map(value)?;
    for (args, value) in map {
        let args = util::get_usize(args)?;
        let value = load_set_from_yaml(value, default.len())?;
        if args >= size {
            return Err(util::YamlContentErr::new(format!(
                "`{}` is greater than the number of the objects for table",
                args,
            )));
        }
        body[args] = value;
    }
    Ok(body)
}

fn load_set_table_2d_from_yaml(
    value: &Yaml,
    size_x: usize,
    size_y: usize,
    default: &Set,
) -> Result<Vec<Vec<Set>>, util::YamlContentErr> {
    let mut body: Vec<Vec<Set>> = (0..size_x)
        .map(|_| (0..size_y).map(|_| default.clone()).collect())
        .collect();
    let map = util::get_map(value)?;
    for (args, value) in map {
        let args = util::get_usize_array(args)?;
        let x = args[0];
        let y = args[1];
        let value = load_set_from_yaml(value, default.len())?;
        if x >= size_x || y >= size_y {
            return Err(util::YamlContentErr::new(format!(
                "`({}, {})` is greater than the numbers of objects for table",
                x, y,
            )));
        }
        body[x][y] = value;
    }
    Ok(body)
}

fn load_set_table_3d_from_yaml(
    value: &Yaml,
    size_x: usize,
    size_y: usize,
    size_z: usize,
    default: &Set,
) -> Result<Vec<Vec<Vec<Set>>>, util::YamlContentErr> {
    let mut body: Vec<Vec<Vec<Set>>> = (0..size_x)
        .map(|_| {
            (0..size_y)
                .map(|_| (0..size_z).map(|_| default.clone()).collect())
                .collect()
        })
        .collect();
    let map = util::get_map(value)?;
    for (args, value) in map {
        let args = util::get_usize_array(args)?;
        let x = args[0];
        let y = args[1];
        let z = args[2];
        let value = load_set_from_yaml(value, default.len())?;
        if x >= size_x || y >= size_y || z >= size_z {
            return Err(util::YamlContentErr::new(format!(
                "`({}, {}, {})` is greater than the numbers of objects for table",
                x, y, z,
            )));
        }
        body[x][y][z] = value;
    }
    Ok(body)
}

fn load_set_table_from_yaml(
    value: &Yaml,
    size: Vec<usize>,
    default: Set,
) -> Result<(FxHashMap<Vec<Element>, Set>, Set), util::YamlContentErr> {
    let map = util::get_map(value)?;
    let mut body = FxHashMap::default();
    for (args, value) in map {
        let args = util::get_usize_array(args)?;
        if args.len() != size.len() {
            return Err(util::YamlContentErr::new(format!(
                "expected `{}` arguments for table, but passed `{}`",
                size.len(),
                args.len()
            )));
        }
        let value = load_set_from_yaml(value, default.len())?;
        if args.iter().zip(size.iter()).any(|(a, b)| a >= b) {
            return Err(util::YamlContentErr::new(format!(
                "`{:?}` is greater than the numbers of objects for table",
                args,
            )));
        }
        body.insert(args, value);
    }
    Ok((body, default))
}

fn load_set_dictionary_from_yaml(
    value: &Yaml,
    default: Set,
) -> Result<(FxHashMap<Vec<Element>, Set>, Set), util::YamlContentErr> {
    let map = util::get_map(value)?;
    let mut body = FxHashMap::default();

    for (args, value) in map {
        let args = util::get_usize_array(args)?;
        let value = load_set_from_yaml(value, default.len())?;
        body.insert(args, value);
    }
    Ok((body, default))
}

fn load_vector_from_yaml(value: &Yaml, capacity: usize) -> Result<Vector, util::YamlContentErr> {
    let value = util::get_usize_array(value)?;
    for v in &value {
        if *v >= capacity {
            return Err(util::YamlContentErr::new(format!(
                "element `{}` in a vector table is too large for the object",
                *v,
            )));
        }
    }
    Ok(value)
}

fn load_vector_table_1d_from_yaml(
    value: &Yaml,
    size: usize,
    default: &[Element],
    capacity: usize,
) -> Result<Vec<Vector>, util::YamlContentErr> {
    let mut body: Vec<Vector> = (0..size).map(|_| default.to_vec()).collect();
    let map = util::get_map(value)?;
    for (args, value) in map {
        let args = util::get_usize(args)?;
        let value = load_vector_from_yaml(value, capacity)?;
        if args >= size {
            return Err(util::YamlContentErr::new(format!(
                "`{}` is greater than the number of the objects for table",
                args,
            )));
        }
        body[args] = value;
    }
    Ok(body)
}

fn load_vector_table_2d_from_yaml(
    value: &Yaml,
    size_x: usize,
    size_y: usize,
    default: &[Element],
    capacity: usize,
) -> Result<Vec<Vec<Vector>>, util::YamlContentErr> {
    let mut body: Vec<Vec<Vector>> = (0..size_x)
        .map(|_| (0..size_y).map(|_| default.to_vec()).collect())
        .collect();
    let map = util::get_map(value)?;
    for (args, value) in map {
        let args = util::get_usize_array(args)?;
        let x = args[0];
        let y = args[1];
        let value = load_vector_from_yaml(value, capacity)?;
        if x >= size_x || y >= size_y {
            return Err(util::YamlContentErr::new(format!(
                "`({}, {})` is greater than the numbers of objects for table",
                x, y,
            )));
        }
        body[x][y] = value;
    }
    Ok(body)
}

fn load_vector_table_3d_from_yaml(
    value: &Yaml,
    size_x: usize,
    size_y: usize,
    size_z: usize,
    default: &[Element],
    capacity: usize,
) -> Result<Vec<Vec<Vec<Vector>>>, util::YamlContentErr> {
    let mut body: Vec<Vec<Vec<Vector>>> = (0..size_x)
        .map(|_| {
            (0..size_y)
                .map(|_| (0..size_z).map(|_| default.to_vec()).collect())
                .collect()
        })
        .collect();
    let map = util::get_map(value)?;
    for (args, value) in map {
        let args = util::get_usize_array(args)?;
        let x = args[0];
        let y = args[1];
        let z = args[2];
        let value = load_vector_from_yaml(value, capacity)?;
        if x >= size_x || y >= size_y || z >= size_z {
            return Err(util::YamlContentErr::new(format!(
                "`({}, {}, {})` is greater than the numbers of objects for table",
                x, y, z,
            )));
        }
        body[x][y][z] = value;
    }
    Ok(body)
}

fn load_vector_table_from_yaml(
    value: &Yaml,
    size: Vec<usize>,
    default: Vector,
    capacity: usize,
) -> Result<(FxHashMap<Vec<Element>, Vector>, Vector), util::YamlContentErr> {
    let map = util::get_map(value)?;
    let mut body = FxHashMap::default();

    for (args, value) in map {
        let args = util::get_usize_array(args)?;
        if args.len() != size.len() {
            return Err(util::YamlContentErr::new(format!(
                "expected `{}` arguments for table, but passed `{}`",
                size.len(),
                args.len()
            )));
        }
        let value = load_vector_from_yaml(value, capacity)?;
        if args.iter().zip(size.iter()).any(|(a, b)| a >= b) {
            return Err(util::YamlContentErr::new(format!(
                "`{:?}` is greater than the numbers of objects for table",
                args,
            )));
        }
        body.insert(args, value);
    }
    Ok((body, default))
}

fn load_vector_dictionary_from_yaml(
    value: &Yaml,
    default: Vector,
    capacity: usize,
) -> Result<(FxHashMap<Vec<Element>, Vector>, Vector), util::YamlContentErr> {
    let map = util::get_map(value)?;
    let mut body = FxHashMap::default();

    for (args, value) in map {
        let args = util::get_usize_array(args)?;
        let value = load_vector_from_yaml(value, capacity)?;
        body.insert(args, value);
    }
    Ok((body, default))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use dypdl::{Table1DHandle, Table2DHandle, Table3DHandle, TableHandle};
    use yaml_rust::yaml::{Array, Hash};
    extern crate lazy_static;

    #[test]
    fn load_table_from_yaml_ok() {
        let mut metadata = StateMetadata::default();
        let ob = metadata.add_object_type(String::from("object"), 3);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let result = metadata.add_element_variable(String::from("e0"), ob);
        assert!(result.is_ok());

        let mut expected = TableRegistry::default();

        expected
            .integer_tables
            .name_to_constant
            .insert(String::from("i0"), 0);
        let result = expected.add_table_1d(String::from("i1"), vec![10, 20, 30]);
        assert!(result.is_ok());
        let result = expected.add_table_2d(
            String::from("i2"),
            vec![vec![10, 20, 30], vec![10, 10, 10], vec![10, 10, 10]],
        );
        assert!(result.is_ok());
        let result = expected.add_table_3d(
            String::from("i3"),
            vec![
                vec![vec![10, 20, 30], vec![0, 0, 0], vec![0, 0, 0]],
                vec![vec![0, 0, 0], vec![0, 0, 0], vec![0, 0, 0]],
                vec![vec![0, 0, 0], vec![0, 0, 0], vec![0, 0, 0]],
            ],
        );
        assert!(result.is_ok());
        let mut map = FxHashMap::default();
        let key = vec![0, 1, 0, 0];
        map.insert(key, 100);
        let key = vec![0, 1, 0, 1];
        map.insert(key, 200);
        let key = vec![0, 1, 2, 0];
        map.insert(key, 300);
        let key = vec![0, 1, 2, 1];
        map.insert(key, 400);
        let result = expected.add_table(String::from("i4"), map, 0);
        assert!(result.is_ok());

        expected
            .continuous_tables
            .name_to_constant
            .insert(String::from("c0"), 0.0);
        let result = expected.add_table_1d(String::from("c1"), vec![10.0, 20.0, 30.0]);
        assert!(result.is_ok());
        let result = expected.add_table_2d(
            String::from("c2"),
            vec![
                vec![10.0, 20.0, 30.0],
                vec![10.0, 10.0, 10.0],
                vec![10.0, 10.0, 10.0],
            ],
        );
        assert!(result.is_ok());
        let result = expected.add_table_3d(
            String::from("c3"),
            vec![
                vec![
                    vec![10.0, 20.0, 30.0],
                    vec![0.0, 0.0, 0.0],
                    vec![0.0, 0.0, 0.0],
                ],
                vec![
                    vec![0.0, 0.0, 0.0],
                    vec![0.0, 0.0, 0.0],
                    vec![0.0, 0.0, 0.0],
                ],
                vec![
                    vec![0.0, 0.0, 0.0],
                    vec![0.0, 0.0, 0.0],
                    vec![0.0, 0.0, 0.0],
                ],
            ],
        );
        assert!(result.is_ok());
        let mut map = FxHashMap::default();
        let key = vec![0, 1, 0, 0];
        map.insert(key, 100.0);
        let key = vec![0, 1, 0, 1];
        map.insert(key, 200.0);
        let key = vec![0, 1, 2, 0];
        map.insert(key, 300.0);
        let key = vec![0, 1, 2, 1];
        map.insert(key, 400.0);
        let result = expected.add_table(String::from("c4"), map, 0.0);
        assert!(result.is_ok());

        expected
            .bool_tables
            .name_to_constant
            .insert(String::from("b0"), true);
        let result = expected.add_table_1d(String::from("b1"), vec![true, false, false]);
        assert!(result.is_ok());
        let result = expected.add_table_2d(
            String::from("b2"),
            vec![
                vec![true, false, false],
                vec![false, false, false],
                vec![false, false, false],
            ],
        );
        assert!(result.is_ok());
        let result = expected.add_table_3d(
            String::from("b3"),
            vec![
                vec![
                    vec![true, false, false],
                    vec![false, false, false],
                    vec![false, false, false],
                ],
                vec![
                    vec![true, false, false],
                    vec![false, false, false],
                    vec![false, false, false],
                ],
                vec![
                    vec![true, false, false],
                    vec![false, false, false],
                    vec![false, false, false],
                ],
            ],
        );
        assert!(result.is_ok());
        let mut map = FxHashMap::default();
        let key = vec![0, 1, 0, 0];
        map.insert(key, true);
        let key = vec![0, 1, 0, 1];
        map.insert(key, false);
        let key = vec![0, 1, 2, 0];
        map.insert(key, false);
        let key = vec![0, 1, 2, 1];
        map.insert(key, false);
        let result = expected.add_table(String::from("b4"), map, false);
        assert!(result.is_ok());

        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(2);
        let default = Set::with_capacity(3);
        expected
            .set_tables
            .name_to_constant
            .insert(String::from("s0"), set.clone());
        let result = expected.add_table_1d(
            String::from("s1"),
            vec![set.clone(), default.clone(), default.clone()],
        );
        assert!(result.is_ok());
        let result = expected.add_table_2d(
            String::from("s2"),
            vec![
                vec![set.clone(), default.clone(), default.clone()],
                vec![default.clone(), default.clone(), default.clone()],
                vec![default.clone(), default.clone(), default.clone()],
            ],
        );
        assert!(result.is_ok());
        let result = expected.add_table_3d(
            String::from("s3"),
            vec![
                vec![
                    vec![set.clone(), default.clone(), default.clone()],
                    vec![default.clone(), default.clone(), default.clone()],
                    vec![default.clone(), default.clone(), default.clone()],
                ],
                vec![
                    vec![set.clone(), default.clone(), default.clone()],
                    vec![default.clone(), default.clone(), default.clone()],
                    vec![default.clone(), default.clone(), default.clone()],
                ],
                vec![
                    vec![set.clone(), default.clone(), default.clone()],
                    vec![default.clone(), default.clone(), default.clone()],
                ],
            ],
        );
        assert!(result.is_ok());
        let mut map = FxHashMap::default();
        let key = vec![0, 1, 0, 0];
        map.insert(key, set);
        let key = vec![0, 1, 0, 1];
        map.insert(key, default.clone());
        let key = vec![0, 1, 2, 0];
        map.insert(key, default.clone());
        let key = vec![0, 1, 2, 1];
        map.insert(key, default.clone());
        let result = expected.add_table(String::from("s4"), map, default);
        assert!(result.is_ok());

        let vector = vec![0, 2];
        let default = Vec::new();
        expected
            .vector_tables
            .name_to_constant
            .insert(String::from("v0"), vector.clone());
        let result = expected.add_table_1d(
            String::from("v1"),
            vec![vector.clone(), default.clone(), default.clone()],
        );
        assert!(result.is_ok());
        let result = expected.add_table_2d(
            String::from("v2"),
            vec![
                vec![vector.clone(), default.clone(), default.clone()],
                vec![default.clone(), default.clone(), default.clone()],
                vec![default.clone(), default.clone(), default.clone()],
            ],
        );
        assert!(result.is_ok());
        let result = expected.add_table_3d(
            String::from("v3"),
            vec![
                vec![
                    vec![vector.clone(), default.clone(), default.clone()],
                    vec![default.clone(), default.clone(), default.clone()],
                    vec![default.clone(), default.clone(), default.clone()],
                ],
                vec![
                    vec![vector.clone(), default.clone(), default.clone()],
                    vec![default.clone(), default.clone(), default.clone()],
                    vec![default.clone(), default.clone(), default.clone()],
                ],
                vec![
                    vec![vector.clone(), default.clone(), default.clone()],
                    vec![default.clone(), default.clone(), default.clone()],
                ],
            ],
        );
        assert!(result.is_ok());
        let mut map = FxHashMap::default();
        let key = vec![0, 1, 0, 0];
        map.insert(key, vector);
        let key = vec![0, 1, 0, 1];
        map.insert(key, default.clone());
        let key = vec![0, 1, 2, 0];
        map.insert(key, default.clone());
        let key = vec![0, 1, 2, 1];
        map.insert(key, default.clone());
        let result = expected.add_table(String::from("v4"), map, default);
        assert!(result.is_ok());

        expected
            .element_tables
            .name_to_constant
            .insert(String::from("t0"), 1);
        let result: Result<Table1DHandle<Element>, _> =
            expected.add_table_1d(String::from("t1"), vec![1, 0, 0]);
        assert!(result.is_ok());
        let result: Result<Table2DHandle<Element>, _> = expected.add_table_2d(
            String::from("t2"),
            vec![vec![1, 0, 0], vec![0, 0, 0], vec![0, 0, 0]],
        );
        assert!(result.is_ok());
        let result: Result<Table3DHandle<Element>, _> = expected.add_table_3d(
            String::from("t3"),
            vec![
                vec![vec![1, 0, 0], vec![0, 0, 0], vec![0, 0, 0]],
                vec![vec![1, 0, 0], vec![0, 0, 0], vec![0, 0, 0]],
                vec![vec![1, 0, 0], vec![0, 0, 0], vec![0, 0, 0]],
            ],
        );
        assert!(result.is_ok());
        let mut map = FxHashMap::default();
        let key = vec![0, 1, 0, 0];
        map.insert(key, 1);
        let key = vec![0, 1, 0, 1];
        map.insert(key, 0);
        let key = vec![0, 1, 2, 0];
        map.insert(key, 0);
        let key = vec![0, 1, 2, 1];
        map.insert(key, 0);
        let result: Result<TableHandle<Element>, _> =
            expected.add_table(String::from("t4"), map, 0);
        assert!(result.is_ok());

        let tables = r"
- name: i0
  type: integer
- name: i1
  type: integer
  args:
        - object
- name: i2
  type: integer 
  args:
        - object
        - object
  default: 10
- name: i3
  type: integer 
  args: [object, object, object]
- name: i4
  type: integer
  args: [object, object, object, object]
- name: c0
  type: continuous
  args: []
- name: c1
  type: continuous
  args:
        - object
- name: c2
  type: continuous 
  args:
        - object
        - object
  default: 10
- name: c3
  type: continuous 
  args: [object, object, object]
- name: c4
  type: continuous
  args: [object, object, object, object]
- name: b0
  type: bool
- name: b1
  type: bool
  args: [object]
- name: b2
  type: bool 
  args: [object, object]
- name: b3
  type: bool
  args:
        - object
        - object
        - object
  default: false
- name: b4
  type: bool
  args:
        - object
        - object
        - object
        - object
- name: s0
  type: set
  object: object
- name: s1
  type: set
  object: object
  args: [object]
  default: []
- name: s2
  type: set
  object: object
  args: [object, object]
- name: s3
  type: set
  object: object
  args: [object, object, object]
- name: s4
  type: set
  object: object
  args: [object, object, object, object]
- name: v0
  type: vector
  object: object
- name: v1
  type: vector
  object: object
  args: [object]
  default: []
- name: v2
  type: vector
  object: object
  args: [object, object]
- name: v3
  type: vector
  object: object
  args: [object, object, object]
- name: v4
  type: vector 
  object: object
  args: [object, object, object, object]
- name: t0
  type: element
- name: t1
  type: element
  args:
        - object
- name: t2
  type: element
  args:
        - object
        - object
  default: 10
- name: t3
  type: element 
  args: [object, object, object]
- name: t4
  type: element
  args: [object, object, object, object]
";
        let table_values = r"
i0: 0
i1:
      0: 10
      1: 20
      2: 30
i2: { [0, 0]: 10, [0, 1]: 20, [0, 2]: 30 }
i3: { [0, 0, 0]: 10, [0, 0, 1]: 20, [0, 0, 2]: 30 }
i4: { [0, 1, 0, 0]: 100, [0, 1, 0, 1]: 200, [0, 1, 2, 0]: 300, [0, 1, 2, 1]: 400 }
c0: 0
c1:
      0: 10
      1: 20
      2: 30
c2: { [0, 0]: 10, [0, 1]: 20, [0, 2]: 30 }
c3: { [0, 0, 0]: 10, [0, 0, 1]: 20, [0, 0, 2]: 30 }
c4: { [0, 1, 0, 0]: 100, [0, 1, 0, 1]: 200, [0, 1, 2, 0]: 300, [0, 1, 2, 1]: 400 }
b0: true
b1: { 0: true, 1: false, 2: false }
b2: { [0, 0]: true }
b3: { [0, 0, 0]: true, [1, 0, 0]: true, [2, 0, 0]: true }
b4: { [0, 1, 0, 0]: true, [0, 1, 0, 1]: false, [0, 1, 2, 0]: false, [0, 1, 2, 1]: false }
s0: [0, 2]
s1: { 0: [0, 2] }
s2: { [0, 0]: [0, 2] }
s3: { [0, 0, 0]: [0, 2], [1, 0, 0]: [0, 2], [2, 0, 0]: [0, 2] }
s4: { [0, 1, 0, 0]: [0, 2]}
v0: [0, 2]
v1: { 0: [0, 2] }
v2: { [0, 0]: [0, 2] }
v3: { [0, 0, 0]: [0, 2], [1, 0, 0]: [0, 2], [2, 0, 0]: [0, 2] }
v4: { [0, 1, 0, 0]: [0, 2]}
t0: 0
t1: { 0: 1 }
t2: { [0, 0]: 1 }
t3: { [0, 0, 0]: 1, [1, 0, 0]: 1, [2, 0, 0]: 1 }
t4: { [0, 1, 0, 0]: 1 }
";

        let tables = yaml_rust::YamlLoader::load_from_str(tables);
        assert!(tables.is_ok());
        let tables = tables.unwrap();
        assert_eq!(tables.len(), 1);
        let tables = &tables[0];

        let table_values = yaml_rust::YamlLoader::load_from_str(table_values);
        assert!(table_values.is_ok());
        let table_values = table_values.unwrap();
        assert_eq!(table_values.len(), 1);
        let table_values = &table_values[0];

        let empty_array = Yaml::Array(Array::new());
        let empty_hash = Yaml::Hash(Hash::new());
        let registry = load_table_registry_from_yaml(
            tables,
            table_values,
            &empty_array,
            &empty_hash,
            &metadata,
        );
        assert!(registry.is_ok());
        let registry = registry.unwrap();
        assert_eq!(registry.integer_tables, expected.integer_tables);
        assert_relative_eq!(registry.continuous_tables, expected.continuous_tables);
        assert_eq!(registry.bool_tables, expected.bool_tables);
    }

    #[test]
    fn load_combined_from_yaml_ok() {
        let mut metadata = StateMetadata::default();
        let ob = metadata.add_object_type(String::from("object"), 3);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let result = metadata.add_element_variable(String::from("e0"), ob);
        assert!(result.is_ok());

        let mut expected = TableRegistry::default();

        expected
            .integer_tables
            .name_to_constant
            .insert(String::from("i0"), 0);
        let result = expected.add_table_1d(String::from("i1"), vec![10, 20, 30]);
        assert!(result.is_ok());
        let mut map = FxHashMap::default();
        let key = vec![0, 1, 0, 0];
        map.insert(key, 100);
        let key = vec![0, 1, 0, 1];
        map.insert(key, 200);
        let key = vec![0, 1, 2, 0];
        map.insert(key, 300);
        let key = vec![0, 1, 2, 1];
        map.insert(key, 400);
        let result = expected.add_table(String::from("i_dic"), map, 0);
        assert!(result.is_ok());

        expected
            .continuous_tables
            .name_to_constant
            .insert(String::from("c0"), 0.0);
        let result = expected.add_table_1d(String::from("c1"), vec![10.0, 20.0, 30.0]);
        assert!(result.is_ok());

        let mut map = FxHashMap::default();
        let key = vec![0, 1, 0, 0];
        map.insert(key, 100.0);
        let key = vec![0, 1, 0, 1];
        map.insert(key, 200.0);
        let key = vec![0, 1, 2, 0];
        map.insert(key, 300.0);
        let key = vec![0, 1, 2, 1];
        map.insert(key, 400.0);
        let result = expected.add_table(String::from("c_dic"), map, 0.0);
        assert!(result.is_ok());

        expected
            .bool_tables
            .name_to_constant
            .insert(String::from("b0"), true);
        let result = expected.add_table_1d(String::from("b1"), vec![true, false, false]);
        assert!(result.is_ok());
        let mut map = FxHashMap::default();
        let key = vec![0, 1, 0, 0];
        map.insert(key, true);
        let key = vec![0, 1, 0, 1];
        map.insert(key, false);
        let key = vec![0, 1, 2, 0];
        map.insert(key, false);
        let key = vec![0, 1, 2, 1];
        map.insert(key, false);
        let result = expected.add_table(String::from("b_dic"), map, false);
        assert!(result.is_ok());

        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(2);
        let default = Set::with_capacity(3);
        expected
            .set_tables
            .name_to_constant
            .insert(String::from("s0"), set.clone());
        let result = expected.add_table_1d(
            String::from("s1"),
            vec![set.clone(), default.clone(), default.clone()],
        );
        assert!(result.is_ok());
        let mut map = FxHashMap::default();
        let key = vec![0, 1, 0, 0];
        map.insert(key, set);
        let key = vec![0, 1, 0, 1];
        map.insert(key, default.clone());
        let key = vec![0, 1, 2, 0];
        map.insert(key, default.clone());
        let key = vec![0, 1, 2, 1];
        map.insert(key, default.clone());
        let result = expected.add_table(String::from("s_dic"), map, default);
        assert!(result.is_ok());

        expected
            .element_tables
            .name_to_constant
            .insert(String::from("t0"), 1);
        let result: Result<Table1DHandle<Element>, _> =
            expected.add_table_1d(String::from("t1"), vec![1, 0, 0]);
        assert!(result.is_ok());
        let mut map = FxHashMap::default();
        let key = vec![0, 1, 0, 0];
        map.insert(key, 1);
        let key = vec![0, 1, 0, 1];
        map.insert(key, 0);
        let key = vec![0, 1, 2, 0];
        map.insert(key, 0);
        let key = vec![0, 1, 2, 1];
        map.insert(key, 0);
        let result: Result<TableHandle<Element>, _> =
            expected.add_table(String::from("t_dic"), map, 0);
        assert!(result.is_ok());

        let tables = r"
- name: i0
  type: integer
- name: i1
  type: integer
  args:
        - object
- name: c0
  type: continuous
  args: []
- name: c1
  type: continuous
  args:
        - 3
- name: b0
  type: bool
- name: b1
  type: bool
  args: [3]
- name: s0
  type: set
  object: object
- name: s1
  type: set
  object: 3
  args: [object]
  default: []
- name: t0
  type: element
- name: t1
  type: element
  args:
        - object
";

        let dictionaries = r"
- name: c_dic
  type: continuous
  default: 0.0
- name: i_dic
  type: integer
  default: 0
- name: s_dic
  type: set
  object: object
  default: []
- name: b_dic
  type: bool
  default: false
- name: t_dic
  type: element
  default: 0
";

        let table_values = r"
i0: 0
i1:
      0: 10
      1: 20
      2: 30
c0: 0
c1:
      0: 10
      1: 20
      2: 30
b0: true
b1: { 0: true, 1: false, 2: false }
s0: [0, 2]
s1: { 0: [0, 2] }
t0: 0
t1: { 0: 1 }
";

        let dictionary_values = r"
i_dic: { [0, 1, 0, 0]: 100, [0, 1, 0, 1]: 200, [0, 1, 2, 0]: 300, [0, 1, 2, 1]: 400 }
c_dic: { [0, 1, 0, 0]: 100, [0, 1, 0, 1]: 200, [0, 1, 2, 0]: 300, [0, 1, 2, 1]: 400 }
b_dic: { [0, 1, 0, 0]: true, [0, 1, 0, 1]: false, [0, 1, 2, 0]: false, [0, 1, 2, 1]: false }
s_dic: { [0, 1, 0, 0]: [0, 2]}
t_dic: { [0, 1, 0, 0]: 1 }
";

        let tables = yaml_rust::YamlLoader::load_from_str(tables);
        assert!(tables.is_ok());
        let tables = tables.unwrap();
        assert_eq!(tables.len(), 1);
        let tables = &tables[0];

        let table_values = yaml_rust::YamlLoader::load_from_str(table_values);
        assert!(table_values.is_ok());
        let table_values = table_values.unwrap();
        assert_eq!(table_values.len(), 1);
        let table_values = &table_values[0];

        let dictionaries = yaml_rust::YamlLoader::load_from_str(dictionaries);
        assert!(dictionaries.is_ok());
        let dictionaries = dictionaries.unwrap();
        assert_eq!(dictionaries.len(), 1);
        let dictionaries = &dictionaries[0];

        let dictionary_values = yaml_rust::YamlLoader::load_from_str(dictionary_values);
        assert!(dictionary_values.is_ok());
        let dictionary_values = dictionary_values.unwrap();
        assert_eq!(dictionary_values.len(), 1);
        let dictionary_values = &dictionary_values[0];

        let registry = load_table_registry_from_yaml(
            tables,
            table_values,
            dictionaries,
            dictionary_values,
            &metadata,
        );
        assert!(registry.is_ok());
        let registry = registry.unwrap();
        assert_eq!(registry.integer_tables, expected.integer_tables);
        assert_relative_eq!(registry.continuous_tables, expected.continuous_tables);
        assert_eq!(registry.bool_tables, expected.bool_tables);
    }

    #[test]
    fn load_dictionaries_from_yaml_ok() {
        let mut metadata = StateMetadata::default();
        let ob = metadata.add_object_type(String::from("object"), 3);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let result = metadata.add_element_variable(String::from("e0"), ob);
        assert!(result.is_ok());

        let mut expected = TableRegistry::default();

        let mut map = FxHashMap::default();
        let key = vec![0, 1, 0, 0];
        map.insert(key, 100);
        let key = vec![0, 1, 0, 1];
        map.insert(key, 200);
        let key = vec![0, 1, 2, 0];
        map.insert(key, 300);
        let key = vec![0, 1, 2, 1];
        map.insert(key, 400);
        let result = expected.add_table(String::from("i_dic"), map, 0);
        assert!(result.is_ok());

        let mut map = FxHashMap::default();
        let key = vec![0, 1, 0, 0];
        map.insert(key, 100.0);
        let key = vec![0, 1, 0, 1];
        map.insert(key, 200.0);
        let key = vec![0, 1, 2, 0];
        map.insert(key, 300.0);
        let key = vec![0, 1, 2, 1];
        map.insert(key, 400.0);
        let result = expected.add_table(String::from("c_dic"), map, 0.0);
        assert!(result.is_ok());

        let mut map = FxHashMap::default();
        let key = vec![0, 1, 0, 0];
        map.insert(key, true);
        let key = vec![0, 1, 0, 1];
        map.insert(key, false);
        let key = vec![0, 1, 2, 0];
        map.insert(key, false);
        let key = vec![0, 1, 2, 1];
        map.insert(key, false);
        let result = expected.add_table(String::from("b_dic"), map, false);
        assert!(result.is_ok());

        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(2);
        let default = Set::with_capacity(3);

        let mut map = FxHashMap::default();
        let key = vec![0, 1, 0, 0];
        map.insert(key, set);
        let key = vec![0, 1, 0, 1];
        map.insert(key, default.clone());
        let key = vec![0, 1, 2, 0];
        map.insert(key, default.clone());
        let key = vec![0, 1, 2, 1];
        map.insert(key, default.clone());
        let result = expected.add_table(String::from("s_dic"), map, default);
        assert!(result.is_ok());

        let mut map = FxHashMap::default();
        let key = vec![0, 1, 0, 0];
        map.insert(key, 1);
        let key = vec![0, 1, 0, 1];
        map.insert(key, 0);
        let key = vec![0, 1, 2, 0];
        map.insert(key, 0);
        let key = vec![0, 1, 2, 1];
        map.insert(key, 0);
        let result: Result<TableHandle<Element>, _> =
            expected.add_table(String::from("t_dic"), map, 0);
        assert!(result.is_ok());

        let dictionaries = r"
- name: c_dic
  type: continuous
  default: 0.0
- name: i_dic
  type: integer
  default: 0
- name: s_dic
  type: set
  object: object
  default: []
- name: b_dic
  type: bool
  default: false
- name: t_dic
  type: element
  default: 0
";
        let dictionary_values = r"
i_dic: { [0, 1, 0, 0]: 100, [0, 1, 0, 1]: 200, [0, 1, 2, 0]: 300, [0, 1, 2, 1]: 400 }
c_dic: { [0, 1, 0, 0]: 100, [0, 1, 0, 1]: 200, [0, 1, 2, 0]: 300, [0, 1, 2, 1]: 400 }
b_dic: { [0, 1, 0, 0]: true, [0, 1, 0, 1]: false, [0, 1, 2, 0]: false, [0, 1, 2, 1]: false }
s_dic: { [0, 1, 0, 0]: [0, 2], [0, 1, 0, 1]: [], [0, 1, 2, 0]: [], [0, 1, 2, 1]: [] }
t_dic: { [0, 1, 0, 0]: 1, [0, 1, 0, 1]: 0, [0, 1, 2, 0]: 0, [0, 1, 2, 1]: 0 }
";

        let dictionaries = yaml_rust::YamlLoader::load_from_str(dictionaries);
        assert!(dictionaries.is_ok());
        let dictionaries = dictionaries.unwrap();
        assert_eq!(dictionaries.len(), 1);
        let dictionaries = &dictionaries[0];

        let dictionary_values = yaml_rust::YamlLoader::load_from_str(dictionary_values);
        assert!(dictionary_values.is_ok());
        let dictionary_values = dictionary_values.unwrap();
        assert_eq!(dictionary_values.len(), 1);
        let dictionary_values = &dictionary_values[0];

        let empty_array = Yaml::Array(Array::new());
        let empty_hash = Yaml::Hash(Hash::new());
        let registry = load_table_registry_from_yaml(
            &empty_array,
            &empty_hash,
            dictionaries,
            dictionary_values,
            &metadata,
        );
        assert!(registry.is_ok());
        let registry = registry.unwrap();
        assert_eq!(registry.integer_tables, expected.integer_tables);
        assert_relative_eq!(registry.continuous_tables, expected.continuous_tables);
        assert_eq!(registry.bool_tables, expected.bool_tables);
        assert_eq!(registry.set_tables, expected.set_tables);
        assert_eq!(registry.element_tables, expected.element_tables);
    }

    #[test]
    fn load_tables_from_yaml_err() {
        let mut metadata = StateMetadata::default();
        let ob = metadata.add_object_type(String::from("object"), 3);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = metadata.add_element_variable(String::from("e0"), ob);
        assert!(v.is_ok());

        let tables = r"
- name: f0
  type: integer
- name: f0
  type: integer
  args: [object]
";

        let tables = yaml_rust::YamlLoader::load_from_str(tables);
        assert!(tables.is_ok());
        let tables = tables.unwrap();
        assert_eq!(tables.len(), 1);
        let tables = &tables[0];

        let table_values = r"
f0: 0
f0:
      0: 10
      1: 20
      2: 30
";
        let table_values = yaml_rust::YamlLoader::load_from_str(table_values);
        assert!(table_values.is_ok());
        let table_values = table_values.unwrap();
        assert_eq!(table_values.len(), 1);
        let table_values = &table_values[0];

        let empty_array = Yaml::Array(Array::new());
        let empty_hash = Yaml::Hash(Hash::new());
        let registry = load_table_registry_from_yaml(
            tables,
            table_values,
            &empty_array,
            &empty_hash,
            &metadata,
        );
        assert!(registry.is_err());

        let tables = r"
- name: object
  type: integer
";

        let tables = yaml_rust::YamlLoader::load_from_str(tables);
        assert!(tables.is_ok());
        let tables = tables.unwrap();
        assert_eq!(tables.len(), 1);
        let tables = &tables[0];

        let table_values = r"
object: 0
";
        let table_values = yaml_rust::YamlLoader::load_from_str(table_values);
        assert!(table_values.is_ok());
        let table_values = table_values.unwrap();
        assert_eq!(table_values.len(), 1);
        let table_values = &table_values[0];

        let registry = load_table_registry_from_yaml(
            tables,
            table_values,
            &empty_array,
            &empty_hash,
            &metadata,
        );
        assert!(registry.is_err());

        let tables = r"
- name: e0
  type: integer
";

        let tables = yaml_rust::YamlLoader::load_from_str(tables);
        assert!(tables.is_ok());
        let tables = tables.unwrap();
        assert_eq!(tables.len(), 1);
        let tables = &tables[0];

        let table_values = r"
e0: 0
";
        let table_values = yaml_rust::YamlLoader::load_from_str(table_values);
        assert!(table_values.is_ok());
        let table_values = table_values.unwrap();
        assert_eq!(table_values.len(), 1);
        let table_values = &table_values[0];

        let registry = load_table_registry_from_yaml(
            tables,
            table_values,
            &empty_array,
            &empty_hash,
            &metadata,
        );
        assert!(registry.is_err());

        let tables = r"
- name: f1
  type: integer
  args: [null]
";

        let tables = yaml_rust::YamlLoader::load_from_str(tables);
        assert!(tables.is_ok());
        let tables = tables.unwrap();
        assert_eq!(tables.len(), 1);
        let tables = &tables[0];

        let table_values = r"
f1:
      0: 10
      1: 20
      2: 30
";
        let table_values = yaml_rust::YamlLoader::load_from_str(table_values);
        assert!(table_values.is_ok());
        let table_values = table_values.unwrap();
        assert_eq!(table_values.len(), 1);
        let table_values = &table_values[0];

        let registry = load_table_registry_from_yaml(
            tables,
            table_values,
            &empty_array,
            &empty_hash,
            &metadata,
        );
        assert!(registry.is_err());

        let tables = r"
- name: f1
  type: null
  args: [object]
";
        let tables = yaml_rust::YamlLoader::load_from_str(tables);
        assert!(tables.is_ok());
        let tables = tables.unwrap();
        assert_eq!(tables.len(), 1);
        let tables = &tables[0];
        let registry = load_table_registry_from_yaml(
            tables,
            table_values,
            &empty_array,
            &empty_hash,
            &metadata,
        );
        assert!(registry.is_err());

        let tables = r"
- name: f1
  type: integer
";
        let tables = yaml_rust::YamlLoader::load_from_str(tables);
        assert!(tables.is_ok());
        let tables = tables.unwrap();
        assert_eq!(tables.len(), 1);
        let tables = &tables[0];
        let registry = load_table_registry_from_yaml(
            tables,
            table_values,
            &empty_array,
            &empty_hash,
            &metadata,
        );
        assert!(registry.is_err());

        let tables = r"
- name: f1
  args: [object]
";
        let tables = yaml_rust::YamlLoader::load_from_str(tables);
        assert!(tables.is_ok());
        let tables = tables.unwrap();
        assert_eq!(tables.len(), 1);
        let tables = &tables[0];
        let registry = load_table_registry_from_yaml(
            tables,
            table_values,
            &empty_array,
            &empty_hash,
            &metadata,
        );
        assert!(registry.is_err());

        let tables = r"
- type: integer
  args: [object]
";
        let tables = yaml_rust::YamlLoader::load_from_str(tables);
        assert!(tables.is_ok());
        let tables = tables.unwrap();
        assert_eq!(tables.len(), 1);
        let tables = &tables[0];
        let registry = load_table_registry_from_yaml(
            tables,
            table_values,
            &empty_array,
            &empty_hash,
            &metadata,
        );
        assert!(registry.is_err());

        let tables = r"
- name: f1
  type: integer
  args: [object]
";
        let tables = yaml_rust::YamlLoader::load_from_str(tables);
        assert!(tables.is_ok());
        let tables = tables.unwrap();
        assert_eq!(tables.len(), 1);
        let tables = &tables[0];

        let table_values = r"
f1:
      0: 10
      1: 20
      2: 30
      3: 40
";
        let table_values = yaml_rust::YamlLoader::load_from_str(table_values);
        assert!(table_values.is_ok());
        let table_values = table_values.unwrap();
        assert_eq!(table_values.len(), 1);
        let table_values = &table_values[0];
        let registry = load_table_registry_from_yaml(
            tables,
            table_values,
            &empty_array,
            &empty_hash,
            &metadata,
        );
        assert!(registry.is_err());

        let table_values = r"
f2:
      0: 10
      1: 20
      2: 30
";
        let table_values = yaml_rust::YamlLoader::load_from_str(table_values);
        assert!(table_values.is_ok());
        let table_values = table_values.unwrap();
        assert_eq!(table_values.len(), 1);
        let table_values = &table_values[0];
        let registry = load_table_registry_from_yaml(
            tables,
            table_values,
            &empty_array,
            &empty_hash,
            &metadata,
        );
        assert!(registry.is_err());

        let table_values = r"
f1:
      0: 10
      1: 2.1
      2: 30
";
        let table_values = yaml_rust::YamlLoader::load_from_str(table_values);
        assert!(table_values.is_ok());
        let table_values = table_values.unwrap();
        assert_eq!(table_values.len(), 1);
        let table_values = &table_values[0];
        let registry = load_table_registry_from_yaml(
            tables,
            table_values,
            &empty_array,
            &empty_hash,
            &metadata,
        );
        assert!(registry.is_err());

        let table_values = r"
f1:
      [0, 0]: 10
      [0, 1]: 20
      [0, 2]: 30
";
        let table_values = yaml_rust::YamlLoader::load_from_str(table_values);
        assert!(table_values.is_ok());
        let table_values = table_values.unwrap();
        assert_eq!(table_values.len(), 1);
        let table_values = &table_values[0];
        let registry = load_table_registry_from_yaml(
            tables,
            table_values,
            &empty_array,
            &empty_hash,
            &metadata,
        );
        assert!(registry.is_err());

        let tables = r"
- name: c0
  type: continuous
  args: [object]
";
        let tables = yaml_rust::YamlLoader::load_from_str(tables);
        assert!(tables.is_ok());
        let tables = tables.unwrap();
        assert_eq!(tables.len(), 1);
        let tables = &tables[0];

        let table_values = r"
c0:
     0: true
     1: 1.2
     2: 1.5
";
        let table_values = yaml_rust::YamlLoader::load_from_str(table_values);
        assert!(table_values.is_ok());
        let table_values = table_values.unwrap();
        assert_eq!(table_values.len(), 1);
        let table_values = &table_values[0];
        let registry = load_table_registry_from_yaml(
            tables,
            table_values,
            &empty_array,
            &empty_hash,
            &metadata,
        );
        assert!(registry.is_err());

        let tables = r"
- name: b1
  type: bool
  args: [object, object]
";
        let tables = yaml_rust::YamlLoader::load_from_str(tables);
        assert!(tables.is_ok());
        let tables = tables.unwrap();
        assert_eq!(tables.len(), 1);
        let tables = &tables[0];

        let table_values = r"
b1:
      0: true
      1: false
      2: false
";
        let table_values = yaml_rust::YamlLoader::load_from_str(table_values);
        assert!(table_values.is_ok());
        let table_values = table_values.unwrap();
        assert_eq!(table_values.len(), 1);
        let table_values = &table_values[0];
        let registry = load_table_registry_from_yaml(
            tables,
            table_values,
            &empty_array,
            &empty_hash,
            &metadata,
        );
        assert!(registry.is_err());

        let table_values = r"
b1:
      [0, 0]: true
      [0, 1]: 0
      [0, 2]: false
";
        let table_values = yaml_rust::YamlLoader::load_from_str(table_values);
        assert!(table_values.is_ok());
        let table_values = table_values.unwrap();
        assert_eq!(table_values.len(), 1);
        let table_values = &table_values[0];
        let registry = load_table_registry_from_yaml(
            tables,
            table_values,
            &empty_array,
            &empty_hash,
            &metadata,
        );
        assert!(registry.is_err());

        let tables = r"
- name: b1
  type: bool
";
        let tables = yaml_rust::YamlLoader::load_from_str(tables);
        assert!(tables.is_ok());
        let tables = tables.unwrap();
        assert_eq!(tables.len(), 1);
        let tables = &tables[0];

        let table_values = r"
b1:
      [0, 0]: true
      [0, 1]: 0
      [0, 2]: false
";
        let table_values = yaml_rust::YamlLoader::load_from_str(table_values);
        assert!(table_values.is_ok());
        let table_values = table_values.unwrap();
        assert_eq!(table_values.len(), 1);
        let table_values = &table_values[0];
        let registry = load_table_registry_from_yaml(
            tables,
            table_values,
            &empty_array,
            &empty_hash,
            &metadata,
        );
        assert!(registry.is_err());

        let tables = r"
- name: f0
  type: set
  object: object
";

        let tables = yaml_rust::YamlLoader::load_from_str(tables);
        assert!(tables.is_ok());
        let tables = tables.unwrap();
        assert_eq!(tables.len(), 1);
        let tables = &tables[0];

        let table_values = r"
f0: [0, 10]
";
        let table_values = yaml_rust::YamlLoader::load_from_str(table_values);
        assert!(table_values.is_ok());
        let table_values = table_values.unwrap();
        assert_eq!(table_values.len(), 1);
        let table_values = &table_values[0];
        let registry = load_table_registry_from_yaml(
            tables,
            table_values,
            &empty_array,
            &empty_hash,
            &metadata,
        );
        assert!(registry.is_err());

        let tables = r"
- name: f0
  type: set
  object: null
";

        let tables = yaml_rust::YamlLoader::load_from_str(tables);
        assert!(tables.is_ok());
        let tables = tables.unwrap();
        assert_eq!(tables.len(), 1);
        let tables = &tables[0];

        let table_values = r"
f0: [0, 1]
";
        let table_values = yaml_rust::YamlLoader::load_from_str(table_values);
        assert!(table_values.is_ok());
        let table_values = table_values.unwrap();
        assert_eq!(table_values.len(), 1);
        let table_values = &table_values[0];
        let registry = load_table_registry_from_yaml(
            tables,
            table_values,
            &empty_array,
            &empty_hash,
            &metadata,
        );
        assert!(registry.is_err());

        let tables = r"
- name: f0
  type: vector
  object: null
";

        let tables = yaml_rust::YamlLoader::load_from_str(tables);
        assert!(tables.is_ok());
        let tables = tables.unwrap();
        assert_eq!(tables.len(), 1);
        let tables = &tables[0];

        let table_values = r"
f0: [0, 1]
";
        let table_values = yaml_rust::YamlLoader::load_from_str(table_values);
        assert!(table_values.is_ok());
        let table_values = table_values.unwrap();
        assert_eq!(table_values.len(), 1);
        let table_values = &table_values[0];
        let registry = load_table_registry_from_yaml(
            tables,
            table_values,
            &empty_array,
            &empty_hash,
            &metadata,
        );
        assert!(registry.is_err());

        let tables = r"
- name: f0
  type: vector
  object: object
";

        let tables = yaml_rust::YamlLoader::load_from_str(tables);
        assert!(tables.is_ok());
        let tables = tables.unwrap();
        assert_eq!(tables.len(), 1);
        let tables = &tables[0];

        let table_values = r"
f0: [0, 10]
";
        let table_values = yaml_rust::YamlLoader::load_from_str(table_values);
        assert!(table_values.is_ok());
        let table_values = table_values.unwrap();
        assert_eq!(table_values.len(), 1);
        let table_values = &table_values[0];
        let registry = load_table_registry_from_yaml(
            tables,
            table_values,
            &empty_array,
            &empty_hash,
            &metadata,
        );
        assert!(registry.is_err());
    }

    #[test]
    fn load_dictionaries_from_yaml_err() {
        let mut metadata = StateMetadata::default();
        let ob = metadata.add_object_type(String::from("object"), 3);
        assert!(ob.is_ok());

        let empty_array = Yaml::Array(Array::new());
        let empty_hash = Yaml::Hash(Hash::new());

        let dictionaries = r"
- name: i0
  type: integer
  default: 0
- name: i0
  type: integer
  default: 1
";

        let dictionaries = yaml_rust::YamlLoader::load_from_str(dictionaries);
        assert!(dictionaries.is_ok());
        let dictionaries = dictionaries.unwrap();
        assert_eq!(dictionaries.len(), 1);
        let dictionaries = &dictionaries[0];

        let dictionary_values = r"
i0: 
      [0, 1]: 1
      [0, 2]: 2
i0:
      [0]: 10
      [1]: 20
      [2]: 30
";
        let dictionary_values = yaml_rust::YamlLoader::load_from_str(dictionary_values);
        assert!(dictionary_values.is_ok());
        let dictionary_values = dictionary_values.unwrap();
        assert_eq!(dictionary_values.len(), 1);
        let dictionary_values = &dictionary_values[0];

        let registry = load_table_registry_from_yaml(
            &empty_array,
            &empty_hash,
            dictionaries,
            dictionary_values,
            &metadata,
        );
        assert!(registry.is_err());

        let dictionaries = r"
- name: i0  
- name: i1
  type: integer
  default: 1
";

        let dictionaries = yaml_rust::YamlLoader::load_from_str(dictionaries);
        assert!(dictionaries.is_ok());
        let dictionaries = dictionaries.unwrap();
        assert_eq!(dictionaries.len(), 1);
        let dictionaries = &dictionaries[0];

        let dictionary_values = r"
i0: 
      [0, 1]: 1
      [0, 2]: 2
i1:
      [0]: 10
      [1]: 20
      [2]: 30
";
        let dictionary_values = yaml_rust::YamlLoader::load_from_str(dictionary_values);
        assert!(dictionary_values.is_ok());
        let dictionary_values = dictionary_values.unwrap();
        assert_eq!(dictionary_values.len(), 1);
        let dictionary_values = &dictionary_values[0];

        let registry = load_table_registry_from_yaml(
            &empty_array,
            &empty_hash,
            dictionaries,
            dictionary_values,
            &metadata,
        );
        assert!(registry.is_err());

        let dictionaries = r"
- name: i0
  type: integer
  default: 0
- name: c0
  type: continuous
  default: 1.0
";

        let dictionaries = yaml_rust::YamlLoader::load_from_str(dictionaries);
        assert!(dictionaries.is_ok());
        let dictionaries = dictionaries.unwrap();
        assert_eq!(dictionaries.len(), 1);
        let dictionaries = &dictionaries[0];

        let dictionary_values = r"
i0: 
      [0, 1]: 1
      [0, 2]: 2
c0:
      0: 10
      1: 20
      2: 30
";
        let dictionary_values = yaml_rust::YamlLoader::load_from_str(dictionary_values);
        assert!(dictionary_values.is_ok());
        let dictionary_values = dictionary_values.unwrap();
        assert_eq!(dictionary_values.len(), 1);
        let dictionary_values = &dictionary_values[0];

        let registry = load_table_registry_from_yaml(
            &empty_array,
            &empty_hash,
            dictionaries,
            dictionary_values,
            &metadata,
        );
        assert!(registry.is_err());

        let dictionaries = r"
- name: i0
  type: integer
  default: 0
- name: c0
  type: continuous
  default: 1.0
";

        let dictionaries = yaml_rust::YamlLoader::load_from_str(dictionaries);
        assert!(dictionaries.is_ok());
        let dictionaries = dictionaries.unwrap();
        assert_eq!(dictionaries.len(), 1);
        let dictionaries = &dictionaries[0];

        let dictionary_values = r"
i0: 
      [0, 1]: 1
      [0, 2]: 2
";
        let dictionary_values = yaml_rust::YamlLoader::load_from_str(dictionary_values);
        assert!(dictionary_values.is_ok());
        let dictionary_values = dictionary_values.unwrap();
        assert_eq!(dictionary_values.len(), 1);
        let dictionary_values = &dictionary_values[0];

        let registry = load_table_registry_from_yaml(
            &empty_array,
            &empty_hash,
            dictionaries,
            dictionary_values,
            &metadata,
        );
        assert!(registry.is_err());

        let dictionaries = r"
- name: i0
  type: integer
  default: 0
- name: s0
  type: set
  default: []
";

        let dictionaries = yaml_rust::YamlLoader::load_from_str(dictionaries);
        assert!(dictionaries.is_ok());
        let dictionaries = dictionaries.unwrap();
        assert_eq!(dictionaries.len(), 1);
        let dictionaries = &dictionaries[0];

        let dictionary_values = r"
i0: 
      [0, 1]: 1
      [0, 2]: 2
s0:
      [0]: [1]
";
        let dictionary_values = yaml_rust::YamlLoader::load_from_str(dictionary_values);
        assert!(dictionary_values.is_ok());
        let dictionary_values = dictionary_values.unwrap();
        assert_eq!(dictionary_values.len(), 1);
        let dictionary_values = &dictionary_values[0];

        let registry = load_table_registry_from_yaml(
            &empty_array,
            &empty_hash,
            dictionaries,
            dictionary_values,
            &metadata,
        );
        assert!(registry.is_err());

        let dictionaries = r"
- name: i0
  type: integer
  default: 0
- name: c0
  type: continuous
  default: 1.0
";

        let dictionaries = yaml_rust::YamlLoader::load_from_str(dictionaries);
        assert!(dictionaries.is_ok());
        let dictionaries = dictionaries.unwrap();
        assert_eq!(dictionaries.len(), 1);
        let dictionaries = &dictionaries[0];

        let dictionary_values = r"
i0: 
      [0, 1]: 1
      [0, 2]: 2
c0:
      [0]: [1]
";
        let dictionary_values = yaml_rust::YamlLoader::load_from_str(dictionary_values);
        assert!(dictionary_values.is_ok());
        let dictionary_values = dictionary_values.unwrap();
        assert_eq!(dictionary_values.len(), 1);
        let dictionary_values = &dictionary_values[0];

        let registry = load_table_registry_from_yaml(
            &empty_array,
            &empty_hash,
            dictionaries,
            dictionary_values,
            &metadata,
        );
        assert!(registry.is_err());
    }
}
