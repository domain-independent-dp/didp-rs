use crate::state;
use crate::table;
use crate::table_data::TableData;
use crate::variable;
use crate::yaml_util;
use lazy_static::lazy_static;
use std::collections;
use std::fmt;
use std::str;
use yaml_rust::Yaml;

#[derive(Debug, PartialEq, Clone, Default)]
pub struct TableRegistry {
    pub integer_tables: TableData<variable::Integer>,
    pub continuous_tables: TableData<variable::Continuous>,
    pub set_tables: TableData<variable::Set>,
    pub vector_tables: TableData<variable::Vector>,
    pub bool_tables: TableData<bool>,
}

enum TableReturnType {
    Integer(variable::Integer),
    Continuous(variable::Continuous),
    Set(variable::Set),
    Vector(usize, variable::Vector),
    Bool(bool),
}

impl TableRegistry {
    pub fn get_name_set(&self) -> collections::HashSet<String> {
        let mut name_set = collections::HashSet::new();
        name_set.extend(self.integer_tables.get_name_set());
        name_set.extend(self.continuous_tables.get_name_set());
        name_set.extend(self.set_tables.get_name_set());
        name_set.extend(self.vector_tables.get_name_set());
        name_set.extend(self.bool_tables.get_name_set());
        name_set
    }

    pub fn load_from_yaml(
        tables: &Yaml,
        table_values: &Yaml,
        metadata: &state::StateMetadata,
    ) -> Result<TableRegistry, yaml_util::YamlContentErr> {
        lazy_static! {
            static ref ARGS_KEY: yaml_rust::Yaml = yaml_rust::Yaml::from_str("args");
        }
        let tables = yaml_util::get_array(tables)?;
        let mut table_names = Vec::with_capacity(tables.len());
        let mut name_to_signature = collections::HashMap::new();
        let mut reserved_names = metadata.get_name_set();
        for value in tables {
            let map = yaml_util::get_map(value)?;
            let name = yaml_util::get_string_by_key(map, "name")?;
            if let Some(name) = reserved_names.get(&name) {
                return Err(yaml_util::YamlContentErr::new(format!(
                    "table name `{}` is already used",
                    name
                )));
            }
            reserved_names.insert(name.clone());
            let args = match map.get(&ARGS_KEY) {
                Some(value) => yaml_util::get_string_array(value)?,
                None => Vec::new(),
            };
            let mut arg_types = Vec::with_capacity(args.len());
            for object in &args {
                if let Some(value) = metadata.name_to_object.get(object) {
                    arg_types.push(*value);
                } else {
                    return Err(yaml_util::YamlContentErr::new(format!(
                        "no such object `{}`",
                        object
                    )));
                }
            }
            let return_type = yaml_util::get_string_by_key(map, "type")?;
            match &return_type[..] {
                "integer" => {
                    if let Ok(value) = yaml_util::get_numeric_by_key(map, "default") {
                        name_to_signature
                            .insert(name.clone(), (arg_types, TableReturnType::Integer(value)));
                    } else {
                        name_to_signature
                            .insert(name.clone(), (arg_types, TableReturnType::Integer(0)));
                    }
                }
                "continuous" => {
                    if let Ok(value) = yaml_util::get_numeric_by_key(map, "default") {
                        name_to_signature.insert(
                            name.clone(),
                            (arg_types, TableReturnType::Continuous(value)),
                        );
                    } else {
                        name_to_signature
                            .insert(name.clone(), (arg_types, TableReturnType::Continuous(0.0)));
                    }
                }
                "set" => {
                    let object_name = yaml_util::get_string_by_key(map, "object")?;
                    let object = match metadata.name_to_object.get(&object_name) {
                        Some(object) => *object,
                        None => {
                            return Err(yaml_util::YamlContentErr::new(format!(
                                "no such object `{}`",
                                object_name
                            )))
                        }
                    };
                    let n = metadata.object_numbers[object];
                    let mut default = variable::Set::with_capacity(n);
                    if let Ok(array) = yaml_util::get_usize_array_by_key(map, "default") {
                        for v in array {
                            if v >= n {
                                return Err(yaml_util::YamlContentErr::new(format!(
                                    "element `{}` is too large for object `{}`",
                                    v, object_name
                                )));
                            }
                            default.insert(v);
                        }
                    }
                    name_to_signature
                        .insert(name.clone(), (arg_types, TableReturnType::Set(default)));
                }
                "vector" => {
                    let object_name = yaml_util::get_string_by_key(map, "object")?;
                    let object = match metadata.name_to_object.get(&object_name) {
                        Some(object) => *object,
                        None => {
                            return Err(yaml_util::YamlContentErr::new(format!(
                                "no such object `{}`",
                                object_name
                            )))
                        }
                    };
                    let n = metadata.object_numbers[object];
                    let default = match yaml_util::get_usize_array_by_key(map, "default") {
                        Ok(array) => {
                            for v in &array {
                                if *v >= n {
                                    return Err(yaml_util::YamlContentErr::new(format!(
                                        "element `{}` is too large for object `{}`",
                                        *v, object_name
                                    )));
                                }
                            }
                            array
                        }
                        _ => Vec::new(),
                    };
                    name_to_signature.insert(
                        name.clone(),
                        (arg_types, TableReturnType::Vector(n, default)),
                    );
                }
                "bool" => {
                    if let Ok(value) = yaml_util::get_bool_by_key(map, "default") {
                        name_to_signature
                            .insert(name.clone(), (arg_types, TableReturnType::Bool(value)));
                    } else {
                        name_to_signature
                            .insert(name.clone(), (arg_types, TableReturnType::Bool(false)));
                    }
                }
                _ => {
                    return Err(yaml_util::YamlContentErr::new(format!(
                        "no such table type `{}`",
                        return_type
                    )))
                }
            }
            table_names.push(name);
        }
        let mut name_to_integer_constant = collections::HashMap::new();
        let mut integer_tables_1d = Vec::new();
        let mut integer_name_to_table_1d = collections::HashMap::new();
        let mut integer_tables_2d = Vec::new();
        let mut integer_name_to_table_2d = collections::HashMap::new();
        let mut integer_tables_3d = Vec::new();
        let mut integer_name_to_table_3d = collections::HashMap::new();
        let mut integer_tables = Vec::new();
        let mut integer_name_to_table = collections::HashMap::new();
        let mut name_to_continuous_constant = collections::HashMap::new();
        let mut continuous_tables_1d = Vec::new();
        let mut continuous_name_to_table_1d = collections::HashMap::new();
        let mut continuous_tables_2d = Vec::new();
        let mut continuous_name_to_table_2d = collections::HashMap::new();
        let mut continuous_tables_3d = Vec::new();
        let mut continuous_name_to_table_3d = collections::HashMap::new();
        let mut continuous_tables = Vec::new();
        let mut continuous_name_to_table = collections::HashMap::new();
        let mut name_to_set_constant = collections::HashMap::new();
        let mut set_tables_1d = Vec::new();
        let mut set_name_to_table_1d = collections::HashMap::new();
        let mut set_tables_2d = Vec::new();
        let mut set_name_to_table_2d = collections::HashMap::new();
        let mut set_tables_3d = Vec::new();
        let mut set_name_to_table_3d = collections::HashMap::new();
        let mut set_tables = Vec::new();
        let mut set_name_to_table = collections::HashMap::new();
        let mut name_to_vector_constant = collections::HashMap::new();
        let mut vector_tables_1d = Vec::new();
        let mut vector_name_to_table_1d = collections::HashMap::new();
        let mut vector_tables_2d = Vec::new();
        let mut vector_name_to_table_2d = collections::HashMap::new();
        let mut vector_tables_3d = Vec::new();
        let mut vector_name_to_table_3d = collections::HashMap::new();
        let mut vector_tables = Vec::new();
        let mut vector_name_to_table = collections::HashMap::new();
        let mut name_to_bool_constant = collections::HashMap::new();
        let mut bool_tables_1d = Vec::new();
        let mut bool_name_to_table_1d = collections::HashMap::new();
        let mut bool_tables_2d = Vec::new();
        let mut bool_name_to_table_2d = collections::HashMap::new();
        let mut bool_tables_3d = Vec::new();
        let mut bool_name_to_table_3d = collections::HashMap::new();
        let mut bool_tables = Vec::new();
        let mut bool_name_to_table = collections::HashMap::new();
        let table_values = yaml_util::get_map(table_values)?;
        for name in table_names {
            let (arg_types, return_type) = name_to_signature.get(&name).unwrap();
            let value = yaml_util::get_yaml_by_key(table_values, &name)?;
            if arg_types.is_empty() {
                match return_type {
                    TableReturnType::Integer(_) => {
                        name_to_integer_constant.insert(name, yaml_util::get_numeric(value)?);
                    }
                    TableReturnType::Continuous(_) => {
                        name_to_continuous_constant.insert(name, yaml_util::get_numeric(value)?);
                    }
                    TableReturnType::Bool(_) => {
                        name_to_bool_constant.insert(name, yaml_util::get_bool(value)?);
                    }
                    TableReturnType::Set(default) => {
                        let value = Self::load_set_from_yaml(value, default.len())?;
                        name_to_set_constant.insert(name, value);
                    }
                    TableReturnType::Vector(capacity, _) => {
                        let value = Self::load_vector_from_yaml(value, *capacity)?;
                        name_to_vector_constant.insert(name, value);
                    }
                }
            } else if arg_types.len() == 1 {
                let size = metadata.object_numbers[arg_types[0]];
                match return_type {
                    TableReturnType::Integer(default) => {
                        let f = Self::load_numeric_table_1d_from_yaml(value, size, *default)?;
                        integer_name_to_table_1d.insert(name, integer_tables_1d.len());
                        integer_tables_1d.push(f);
                    }
                    TableReturnType::Continuous(default) => {
                        let f = Self::load_numeric_table_1d_from_yaml(value, size, *default)?;
                        continuous_name_to_table_1d.insert(name, continuous_tables_1d.len());
                        continuous_tables_1d.push(f);
                    }
                    TableReturnType::Bool(default) => {
                        let f = Self::load_bool_table_1d_from_yaml(value, size, *default)?;
                        bool_name_to_table_1d.insert(name, bool_tables_1d.len());
                        bool_tables_1d.push(f);
                    }
                    TableReturnType::Set(default) => {
                        let f = Self::load_set_table_1d_from_yaml(value, size, default)?;
                        set_name_to_table_1d.insert(name, set_tables_1d.len());
                        set_tables_1d.push(f);
                    }
                    TableReturnType::Vector(capacity, default) => {
                        let f =
                            Self::load_vector_table_1d_from_yaml(value, size, default, *capacity)?;
                        vector_name_to_table_1d.insert(name, vector_tables_1d.len());
                        vector_tables_1d.push(f);
                    }
                }
            } else if arg_types.len() == 2 {
                let size_x = metadata.object_numbers[arg_types[0]];
                let size_y = metadata.object_numbers[arg_types[1]];
                match return_type {
                    TableReturnType::Integer(default) => {
                        let f =
                            Self::load_numeric_table_2d_from_yaml(value, size_x, size_y, *default)?;
                        integer_name_to_table_2d.insert(name, integer_tables_2d.len());
                        integer_tables_2d.push(f);
                    }
                    TableReturnType::Continuous(default) => {
                        let f =
                            Self::load_numeric_table_2d_from_yaml(value, size_x, size_y, *default)?;
                        continuous_name_to_table_2d.insert(name, continuous_tables_2d.len());
                        continuous_tables_2d.push(f);
                    }
                    TableReturnType::Bool(default) => {
                        let f =
                            Self::load_bool_table_2d_from_yaml(value, size_x, size_y, *default)?;
                        bool_name_to_table_2d.insert(name, bool_tables_2d.len());
                        bool_tables_2d.push(f);
                    }
                    TableReturnType::Set(default) => {
                        let f = Self::load_set_table_2d_from_yaml(value, size_x, size_y, default)?;
                        set_name_to_table_2d.insert(name, set_tables_2d.len());
                        set_tables_2d.push(f);
                    }
                    TableReturnType::Vector(capacity, default) => {
                        let f = Self::load_vector_table_2d_from_yaml(
                            value, size_x, size_y, default, *capacity,
                        )?;
                        vector_name_to_table_2d.insert(name, vector_tables_2d.len());
                        vector_tables_2d.push(f);
                    }
                }
            } else if arg_types.len() == 3 {
                let size_x = metadata.object_numbers[arg_types[0]];
                let size_y = metadata.object_numbers[arg_types[1]];
                let size_z = metadata.object_numbers[arg_types[2]];
                match return_type {
                    TableReturnType::Integer(default) => {
                        let f = Self::load_numeric_table_3d_from_yaml(
                            value, size_x, size_y, size_z, *default,
                        )?;
                        integer_name_to_table_3d.insert(name, integer_tables_3d.len());
                        integer_tables_3d.push(f);
                    }
                    TableReturnType::Continuous(default) => {
                        let f = Self::load_numeric_table_3d_from_yaml(
                            value, size_x, size_y, size_z, *default,
                        )?;
                        continuous_name_to_table_3d.insert(name, continuous_tables_3d.len());
                        continuous_tables_3d.push(f);
                    }
                    TableReturnType::Bool(default) => {
                        let f = Self::load_bool_table_3d_from_yaml(
                            value, size_x, size_y, size_z, *default,
                        )?;
                        bool_name_to_table_3d.insert(name, bool_tables_3d.len());
                        bool_tables_3d.push(f);
                    }
                    TableReturnType::Set(default) => {
                        let f = Self::load_set_table_3d_from_yaml(
                            value, size_x, size_y, size_z, default,
                        )?;
                        set_name_to_table_3d.insert(name, set_tables_3d.len());
                        set_tables_3d.push(f);
                    }
                    TableReturnType::Vector(capacity, default) => {
                        let f = Self::load_vector_table_3d_from_yaml(
                            value, size_x, size_y, size_z, default, *capacity,
                        )?;
                        vector_name_to_table_3d.insert(name, vector_tables_3d.len());
                        vector_tables_3d.push(f);
                    }
                }
            } else {
                let size: Vec<usize> = arg_types
                    .iter()
                    .map(|i| metadata.object_numbers[*i])
                    .collect();
                match return_type {
                    TableReturnType::Integer(default) => {
                        let f = Self::load_numeric_table_from_yaml(value, size, *default)?;
                        integer_name_to_table.insert(name, integer_tables.len());
                        integer_tables.push(f);
                    }
                    TableReturnType::Continuous(default) => {
                        let f = Self::load_numeric_table_from_yaml(value, size, *default)?;
                        continuous_name_to_table.insert(name, continuous_tables.len());
                        continuous_tables.push(f);
                    }
                    TableReturnType::Bool(default) => {
                        let f = Self::load_bool_table_from_yaml(value, size, *default)?;
                        bool_name_to_table.insert(name, bool_tables.len());
                        bool_tables.push(f);
                    }
                    TableReturnType::Set(default) => {
                        let f = Self::load_set_table_from_yaml(value, size, default.clone())?;
                        set_name_to_table.insert(name, set_tables.len());
                        set_tables.push(f);
                    }
                    TableReturnType::Vector(capacity, default) => {
                        let f = Self::load_vector_table_from_yaml(
                            value,
                            size,
                            default.clone(),
                            *capacity,
                        )?;
                        vector_name_to_table.insert(name, vector_tables.len());
                        vector_tables.push(f);
                    }
                }
            }
        }
        Ok(TableRegistry {
            integer_tables: TableData {
                name_to_constant: name_to_integer_constant,
                tables_1d: integer_tables_1d,
                name_to_table_1d: integer_name_to_table_1d,
                tables_2d: integer_tables_2d,
                name_to_table_2d: integer_name_to_table_2d,
                tables_3d: integer_tables_3d,
                name_to_table_3d: integer_name_to_table_3d,
                tables: integer_tables,
                name_to_table: integer_name_to_table,
            },
            continuous_tables: TableData {
                name_to_constant: name_to_continuous_constant,
                tables_1d: continuous_tables_1d,
                name_to_table_1d: continuous_name_to_table_1d,
                tables_2d: continuous_tables_2d,
                name_to_table_2d: continuous_name_to_table_2d,
                tables_3d: continuous_tables_3d,
                name_to_table_3d: continuous_name_to_table_3d,
                tables: continuous_tables,
                name_to_table: continuous_name_to_table,
            },
            set_tables: TableData {
                name_to_constant: name_to_set_constant,
                tables_1d: set_tables_1d,
                name_to_table_1d: set_name_to_table_1d,
                tables_2d: set_tables_2d,
                name_to_table_2d: set_name_to_table_2d,
                tables_3d: set_tables_3d,
                name_to_table_3d: set_name_to_table_3d,
                tables: set_tables,
                name_to_table: set_name_to_table,
            },
            vector_tables: TableData {
                name_to_constant: name_to_vector_constant,
                tables_1d: vector_tables_1d,
                name_to_table_1d: vector_name_to_table_1d,
                tables_2d: vector_tables_2d,
                name_to_table_2d: vector_name_to_table_2d,
                tables_3d: vector_tables_3d,
                name_to_table_3d: vector_name_to_table_3d,
                tables: vector_tables,
                name_to_table: vector_name_to_table,
            },
            bool_tables: TableData {
                name_to_constant: name_to_bool_constant,
                tables_1d: bool_tables_1d,
                name_to_table_1d: bool_name_to_table_1d,
                tables_2d: bool_tables_2d,
                name_to_table_2d: bool_name_to_table_2d,
                tables_3d: bool_tables_3d,
                name_to_table_3d: bool_name_to_table_3d,
                tables: bool_tables,
                name_to_table: bool_name_to_table,
            },
        })
    }

    fn load_numeric_table_1d_from_yaml<T: variable::Numeric>(
        value: &Yaml,
        size: usize,
        default: T,
    ) -> Result<table::Table1D<T>, yaml_util::YamlContentErr>
    where
        <T as str::FromStr>::Err: fmt::Debug,
    {
        let mut body: Vec<T> = (0..size).map(|_| default).collect();
        let map = yaml_util::get_map(value)?;
        for (args, value) in map {
            let args = yaml_util::get_usize(args)?;
            let value = yaml_util::get_numeric(value)?;
            if args >= size {
                return Err(yaml_util::YamlContentErr::new(format!(
                    "`{}` is greater than the number of the objects for table",
                    args,
                )));
            }
            body[args] = value;
        }
        Ok(table::Table1D::new(body))
    }

    fn load_numeric_table_2d_from_yaml<T: variable::Numeric>(
        value: &Yaml,
        size_x: usize,
        size_y: usize,
        default: T,
    ) -> Result<table::Table2D<T>, yaml_util::YamlContentErr>
    where
        <T as str::FromStr>::Err: fmt::Debug,
    {
        let mut body: Vec<Vec<T>> = (0..size_x)
            .map(|_| (0..size_y).map(|_| default).collect())
            .collect();
        let map = yaml_util::get_map(value)?;
        for (args, value) in map {
            let args = yaml_util::get_usize_array(args)?;
            let x = args[0];
            let y = args[1];
            let value = yaml_util::get_numeric(value)?;
            if x >= size_x || y >= size_y {
                return Err(yaml_util::YamlContentErr::new(format!(
                    "`({}, {})` is greater than the numbers of objects for table",
                    x, y,
                )));
            }
            body[x][y] = value;
        }
        Ok(table::Table2D::new(body))
    }

    fn load_numeric_table_3d_from_yaml<T: variable::Numeric>(
        value: &Yaml,
        size_x: usize,
        size_y: usize,
        size_z: usize,
        default: T,
    ) -> Result<table::Table3D<T>, yaml_util::YamlContentErr>
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
        let map = yaml_util::get_map(value)?;
        for (args, value) in map {
            let args = yaml_util::get_usize_array(args)?;
            let x = args[0];
            let y = args[1];
            let z = args[2];
            let value = yaml_util::get_numeric(value)?;
            if x >= size_x || y >= size_y || z >= size_z {
                return Err(yaml_util::YamlContentErr::new(format!(
                    "`({}, {}, {})` is greater than the numbers of objects for table",
                    x, y, z,
                )));
            }
            body[x][y][z] = value;
        }
        Ok(table::Table3D::new(body))
    }

    fn load_numeric_table_from_yaml<T: variable::Numeric>(
        value: &Yaml,
        size: Vec<usize>,
        default: T,
    ) -> Result<table::Table<T>, yaml_util::YamlContentErr>
    where
        <T as str::FromStr>::Err: fmt::Debug,
    {
        let map = yaml_util::get_map(value)?;
        let mut body = collections::HashMap::with_capacity(map.len());
        for (args, value) in map {
            let args = yaml_util::get_usize_array(args)?;
            if args.len() != size.len() {
                return Err(yaml_util::YamlContentErr::new(format!(
                    "expected `{}` arguments for table, but passed `{}`",
                    size.len(),
                    args.len()
                )));
            }
            let value = yaml_util::get_numeric(value)?;
            if args.iter().zip(size.iter()).any(|(a, b)| a >= b) {
                return Err(yaml_util::YamlContentErr::new(format!(
                    "`{:?}` is greater than the numbers of objects for table",
                    args,
                )));
            }
            body.insert(args, value);
        }
        Ok(table::Table::new(body, default))
    }

    fn load_bool_table_1d_from_yaml(
        value: &Yaml,
        size: usize,
        default: bool,
    ) -> Result<table::Table1D<bool>, yaml_util::YamlContentErr> {
        let mut body: Vec<bool> = (0..size).map(|_| default).collect();
        let map = yaml_util::get_map(value)?;
        for (args, value) in map {
            let args = yaml_util::get_usize(args)?;
            let value = yaml_util::get_bool(value)?;
            if args >= size {
                return Err(yaml_util::YamlContentErr::new(format!(
                    "`{}` is greater than the number of the objects for table",
                    args,
                )));
            }
            body[args] = value;
        }
        Ok(table::Table1D::new(body))
    }

    fn load_bool_table_2d_from_yaml(
        value: &Yaml,
        size_x: usize,
        size_y: usize,
        default: bool,
    ) -> Result<table::Table2D<bool>, yaml_util::YamlContentErr> {
        let mut body: Vec<Vec<bool>> = (0..size_x)
            .map(|_| (0..size_y).map(|_| default).collect())
            .collect();
        let map = yaml_util::get_map(value)?;
        for (args, value) in map {
            let args = yaml_util::get_usize_array(args)?;
            let x = args[0];
            let y = args[1];
            let value = yaml_util::get_bool(value)?;
            if x >= size_x || y >= size_y {
                return Err(yaml_util::YamlContentErr::new(format!(
                    "`({}, {})` is greater than the numbers of objects for table",
                    x, y,
                )));
            }
            body[x][y] = value;
        }
        Ok(table::Table2D::new(body))
    }

    fn load_bool_table_3d_from_yaml(
        value: &Yaml,
        size_x: usize,
        size_y: usize,
        size_z: usize,
        default: bool,
    ) -> Result<table::Table3D<bool>, yaml_util::YamlContentErr> {
        let mut body: Vec<Vec<Vec<bool>>> = (0..size_x)
            .map(|_| {
                (0..size_y)
                    .map(|_| (0..size_z).map(|_| default).collect())
                    .collect()
            })
            .collect();
        let map = yaml_util::get_map(value)?;
        for (args, value) in map {
            let args = yaml_util::get_usize_array(args)?;
            let x = args[0];
            let y = args[1];
            let z = args[2];
            let value = yaml_util::get_bool(value)?;
            if x >= size_x || y >= size_y || z >= size_z {
                return Err(yaml_util::YamlContentErr::new(format!(
                    "`({}, {}, {})` is greater than the numbers of objects for table",
                    x, y, z,
                )));
            }
            body[x][y][z] = value;
        }
        Ok(table::Table3D::new(body))
    }

    fn load_bool_table_from_yaml(
        value: &Yaml,
        size: Vec<usize>,
        default: bool,
    ) -> Result<table::Table<bool>, yaml_util::YamlContentErr> {
        let map = yaml_util::get_map(value)?;
        let mut body = collections::HashMap::with_capacity(map.len());
        for (args, value) in map {
            let args = yaml_util::get_usize_array(args)?;
            if args.len() != size.len() {
                return Err(yaml_util::YamlContentErr::new(format!(
                    "expected `{}` arguments for table, but passed `{}`",
                    size.len(),
                    args.len()
                )));
            }
            let value = yaml_util::get_bool(value)?;
            if args.iter().zip(size.iter()).any(|(a, b)| a >= b) {
                return Err(yaml_util::YamlContentErr::new(format!(
                    "`{:?}` is greater than the numbers of objects for table",
                    args,
                )));
            }
            body.insert(args, value);
        }
        Ok(table::Table::new(body, default))
    }

    fn load_set_from_yaml(
        value: &Yaml,
        capacity: usize,
    ) -> Result<variable::Set, yaml_util::YamlContentErr> {
        let array = yaml_util::get_usize_array(value)?;
        let mut set = variable::Set::with_capacity(capacity);
        for v in array {
            if v >= capacity {
                return Err(yaml_util::YamlContentErr::new(format!(
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
        default: &variable::Set,
    ) -> Result<table::Table1D<variable::Set>, yaml_util::YamlContentErr> {
        let mut body: Vec<variable::Set> = (0..size).map(|_| default.clone()).collect();
        let map = yaml_util::get_map(value)?;
        for (args, value) in map {
            let args = yaml_util::get_usize(args)?;
            let value = Self::load_set_from_yaml(value, default.len())?;
            if args >= size {
                return Err(yaml_util::YamlContentErr::new(format!(
                    "`{}` is greater than the number of the objects for table",
                    args,
                )));
            }
            body[args] = value;
        }
        Ok(table::Table1D::new(body))
    }

    fn load_set_table_2d_from_yaml(
        value: &Yaml,
        size_x: usize,
        size_y: usize,
        default: &variable::Set,
    ) -> Result<table::Table2D<variable::Set>, yaml_util::YamlContentErr> {
        let mut body: Vec<Vec<variable::Set>> = (0..size_x)
            .map(|_| (0..size_y).map(|_| default.clone()).collect())
            .collect();
        let map = yaml_util::get_map(value)?;
        for (args, value) in map {
            let args = yaml_util::get_usize_array(args)?;
            let x = args[0];
            let y = args[1];
            let value = Self::load_set_from_yaml(value, default.len())?;
            if x >= size_x || y >= size_y {
                return Err(yaml_util::YamlContentErr::new(format!(
                    "`({}, {})` is greater than the numbers of objects for table",
                    x, y,
                )));
            }
            body[x][y] = value;
        }
        Ok(table::Table2D::new(body))
    }

    fn load_set_table_3d_from_yaml(
        value: &Yaml,
        size_x: usize,
        size_y: usize,
        size_z: usize,
        default: &variable::Set,
    ) -> Result<table::Table3D<variable::Set>, yaml_util::YamlContentErr> {
        let mut body: Vec<Vec<Vec<variable::Set>>> = (0..size_x)
            .map(|_| {
                (0..size_y)
                    .map(|_| (0..size_z).map(|_| default.clone()).collect())
                    .collect()
            })
            .collect();
        let map = yaml_util::get_map(value)?;
        for (args, value) in map {
            let args = yaml_util::get_usize_array(args)?;
            let x = args[0];
            let y = args[1];
            let z = args[2];
            let value = Self::load_set_from_yaml(value, default.len())?;
            if x >= size_x || y >= size_y || z >= size_z {
                return Err(yaml_util::YamlContentErr::new(format!(
                    "`({}, {}, {})` is greater than the numbers of objects for table",
                    x, y, z,
                )));
            }
            body[x][y][z] = value;
        }
        Ok(table::Table3D::new(body))
    }

    fn load_set_table_from_yaml(
        value: &Yaml,
        size: Vec<usize>,
        default: variable::Set,
    ) -> Result<table::Table<variable::Set>, yaml_util::YamlContentErr> {
        let map = yaml_util::get_map(value)?;
        let mut body = collections::HashMap::with_capacity(map.len());
        for (args, value) in map {
            let args = yaml_util::get_usize_array(args)?;
            if args.len() != size.len() {
                return Err(yaml_util::YamlContentErr::new(format!(
                    "expected `{}` arguments for table, but passed `{}`",
                    size.len(),
                    args.len()
                )));
            }
            let value = Self::load_set_from_yaml(value, default.len())?;
            if args.iter().zip(size.iter()).any(|(a, b)| a >= b) {
                return Err(yaml_util::YamlContentErr::new(format!(
                    "`{:?}` is greater than the numbers of objects for table",
                    args,
                )));
            }
            body.insert(args, value);
        }
        Ok(table::Table::new(body, default))
    }

    fn load_vector_from_yaml(
        value: &Yaml,
        capacity: usize,
    ) -> Result<variable::Vector, yaml_util::YamlContentErr> {
        let value = yaml_util::get_usize_array(value)?;
        for v in &value {
            if *v >= capacity {
                return Err(yaml_util::YamlContentErr::new(format!(
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
        default: &[variable::Element],
        capacity: usize,
    ) -> Result<table::Table1D<variable::Vector>, yaml_util::YamlContentErr> {
        let mut body: Vec<variable::Vector> = (0..size).map(|_| default.to_vec()).collect();
        let map = yaml_util::get_map(value)?;
        for (args, value) in map {
            let args = yaml_util::get_usize(args)?;
            let value = Self::load_vector_from_yaml(value, capacity)?;
            if args >= size {
                return Err(yaml_util::YamlContentErr::new(format!(
                    "`{}` is greater than the number of the objects for table",
                    args,
                )));
            }
            body[args] = value;
        }
        Ok(table::Table1D::new(body))
    }

    fn load_vector_table_2d_from_yaml(
        value: &Yaml,
        size_x: usize,
        size_y: usize,
        default: &[variable::Element],
        capacity: usize,
    ) -> Result<table::Table2D<variable::Vector>, yaml_util::YamlContentErr> {
        let mut body: Vec<Vec<variable::Vector>> = (0..size_x)
            .map(|_| (0..size_y).map(|_| default.to_vec()).collect())
            .collect();
        let map = yaml_util::get_map(value)?;
        for (args, value) in map {
            let args = yaml_util::get_usize_array(args)?;
            let x = args[0];
            let y = args[1];
            let value = Self::load_vector_from_yaml(value, capacity)?;
            if x >= size_x || y >= size_y {
                return Err(yaml_util::YamlContentErr::new(format!(
                    "`({}, {})` is greater than the numbers of objects for table",
                    x, y,
                )));
            }
            body[x][y] = value;
        }
        Ok(table::Table2D::new(body))
    }

    fn load_vector_table_3d_from_yaml(
        value: &Yaml,
        size_x: usize,
        size_y: usize,
        size_z: usize,
        default: &[variable::Element],
        capacity: usize,
    ) -> Result<table::Table3D<variable::Vector>, yaml_util::YamlContentErr> {
        let mut body: Vec<Vec<Vec<variable::Vector>>> = (0..size_x)
            .map(|_| {
                (0..size_y)
                    .map(|_| (0..size_z).map(|_| default.to_vec()).collect())
                    .collect()
            })
            .collect();
        let map = yaml_util::get_map(value)?;
        for (args, value) in map {
            let args = yaml_util::get_usize_array(args)?;
            let x = args[0];
            let y = args[1];
            let z = args[2];
            let value = Self::load_vector_from_yaml(value, capacity)?;
            if x >= size_x || y >= size_y || z >= size_z {
                return Err(yaml_util::YamlContentErr::new(format!(
                    "`({}, {}, {})` is greater than the numbers of objects for table",
                    x, y, z,
                )));
            }
            body[x][y][z] = value;
        }
        Ok(table::Table3D::new(body))
    }

    fn load_vector_table_from_yaml(
        value: &Yaml,
        size: Vec<usize>,
        default: variable::Vector,
        capacity: usize,
    ) -> Result<table::Table<variable::Vector>, yaml_util::YamlContentErr> {
        let map = yaml_util::get_map(value)?;
        let mut body = collections::HashMap::with_capacity(map.len());
        for (args, value) in map {
            let args = yaml_util::get_usize_array(args)?;
            if args.len() != size.len() {
                return Err(yaml_util::YamlContentErr::new(format!(
                    "expected `{}` arguments for table, but passed `{}`",
                    size.len(),
                    args.len()
                )));
            }
            let value = Self::load_vector_from_yaml(value, capacity)?;
            if args.iter().zip(size.iter()).any(|(a, b)| a >= b) {
                return Err(yaml_util::YamlContentErr::new(format!(
                    "`{:?}` is greater than the numbers of objects for table",
                    args,
                )));
            }
            body.insert(args, value);
        }
        Ok(table::Table::new(body, default))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use collections::HashMap;

    fn generate_metadata() -> state::StateMetadata {
        let object_names = vec![String::from("object")];
        let mut name_to_object = HashMap::new();
        name_to_object.insert(String::from("object"), 0);
        let object_numbers = vec![3];

        let element_variable_names = vec![String::from("e0")];
        let mut name_to_element_variable = HashMap::new();
        name_to_element_variable.insert(String::from("e0"), 0);
        let element_variable_to_object = vec![0];

        state::StateMetadata {
            object_names,
            name_to_object,
            object_numbers,
            element_variable_names,
            name_to_element_variable,
            element_variable_to_object,
            ..Default::default()
        }
    }

    fn generate_registry() -> TableRegistry {
        let mut name_to_constant = HashMap::new();
        name_to_constant.insert(String::from("i0"), 0);

        let tables_1d = vec![table::Table1D::new(vec![10, 20, 30])];
        let mut name_to_table_1d = HashMap::new();
        name_to_table_1d.insert(String::from("i1"), 0);

        let tables_2d = vec![table::Table2D::new(vec![
            vec![10, 20, 30],
            vec![10, 10, 10],
            vec![10, 10, 10],
        ])];
        let mut name_to_table_2d = HashMap::new();
        name_to_table_2d.insert(String::from("i2"), 0);

        let tables_3d = vec![table::Table3D::new(vec![
            vec![vec![10, 20, 30], vec![0, 0, 0], vec![0, 0, 0]],
            vec![vec![0, 0, 0], vec![0, 0, 0], vec![0, 0, 0]],
            vec![vec![0, 0, 0], vec![0, 0, 0], vec![0, 0, 0]],
        ])];
        let mut name_to_table_3d = HashMap::new();
        name_to_table_3d.insert(String::from("i3"), 0);

        let mut map = HashMap::new();
        let key = vec![0, 1, 0, 0];
        map.insert(key, 100);
        let key = vec![0, 1, 0, 1];
        map.insert(key, 200);
        let key = vec![0, 1, 2, 0];
        map.insert(key, 300);
        let key = vec![0, 1, 2, 1];
        map.insert(key, 400);
        let tables = vec![table::Table::new(map, 0)];
        let mut name_to_table = HashMap::new();
        name_to_table.insert(String::from("i4"), 0);

        let integer_tables = TableData {
            name_to_constant,
            tables_1d,
            name_to_table_1d,
            tables_2d,
            name_to_table_2d,
            tables_3d,
            name_to_table_3d,
            tables,
            name_to_table,
        };

        let mut name_to_constant = HashMap::new();
        name_to_constant.insert(String::from("c0"), 0.0);

        let tables_1d = vec![table::Table1D::new(vec![10.0, 20.0, 30.0])];
        let mut name_to_table_1d = HashMap::new();
        name_to_table_1d.insert(String::from("c1"), 0);

        let tables_2d = vec![table::Table2D::new(vec![
            vec![10.0, 20.0, 30.0],
            vec![10.0, 10.0, 10.0],
            vec![10.0, 10.0, 10.0],
        ])];
        let mut name_to_table_2d = HashMap::new();
        name_to_table_2d.insert(String::from("c2"), 0);

        let tables_3d = vec![table::Table3D::new(vec![
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
        ])];
        let mut name_to_table_3d = HashMap::new();
        name_to_table_3d.insert(String::from("c3"), 0);

        let mut map = HashMap::new();
        let key = vec![0, 1, 0, 0];
        map.insert(key, 100.0);
        let key = vec![0, 1, 0, 1];
        map.insert(key, 200.0);
        let key = vec![0, 1, 2, 0];
        map.insert(key, 300.0);
        let key = vec![0, 1, 2, 1];
        map.insert(key, 400.0);
        let tables = vec![table::Table::new(map, 0.0)];
        let mut name_to_table = HashMap::new();
        name_to_table.insert(String::from("c4"), 0);

        let continuous_tables = TableData {
            name_to_constant,
            tables_1d,
            name_to_table_1d,
            tables_2d,
            name_to_table_2d,
            tables_3d,
            name_to_table_3d,
            tables,
            name_to_table,
        };

        let mut name_to_constant = HashMap::new();
        name_to_constant.insert(String::from("b0"), true);

        let tables_1d = vec![table::Table1D::new(vec![true, false, false])];
        let mut name_to_table_1d = HashMap::new();
        name_to_table_1d.insert(String::from("b1"), 0);

        let tables_2d = vec![table::Table2D::new(vec![
            vec![true, false, false],
            vec![false, false, false],
            vec![false, false, false],
        ])];
        let mut name_to_table_2d = HashMap::new();
        name_to_table_2d.insert(String::from("b2"), 0);

        let tables_3d = vec![table::Table3D::new(vec![
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
        ])];
        let mut name_to_table_3d = HashMap::new();
        name_to_table_3d.insert(String::from("b3"), 0);

        let mut map = HashMap::new();
        let key = vec![0, 1, 0, 0];
        map.insert(key, true);
        let key = vec![0, 1, 0, 1];
        map.insert(key, false);
        let key = vec![0, 1, 2, 0];
        map.insert(key, false);
        let key = vec![0, 1, 2, 1];
        map.insert(key, false);
        let tables = vec![table::Table::new(map, false)];
        let mut name_to_table = HashMap::new();
        name_to_table.insert(String::from("b4"), 0);

        let bool_tables = TableData {
            name_to_constant,
            tables_1d,
            name_to_table_1d,
            tables_2d,
            name_to_table_2d,
            tables_3d,
            name_to_table_3d,
            tables,
            name_to_table,
        };

        let mut name_to_constant = HashMap::new();
        let mut set = variable::Set::with_capacity(3);
        set.insert(0);
        set.insert(2);
        let default = variable::Set::with_capacity(3);
        name_to_constant.insert(String::from("s0"), set.clone());

        let tables_1d = vec![table::Table1D::new(vec![
            set.clone(),
            default.clone(),
            default.clone(),
        ])];
        let mut name_to_table_1d = HashMap::new();
        name_to_table_1d.insert(String::from("s1"), 0);

        let tables_2d = vec![table::Table2D::new(vec![
            vec![set.clone(), default.clone(), default.clone()],
            vec![default.clone(), default.clone(), default.clone()],
            vec![default.clone(), default.clone(), default.clone()],
        ])];
        let mut name_to_table_2d = HashMap::new();
        name_to_table_2d.insert(String::from("s2"), 0);

        let tables_3d = vec![table::Table3D::new(vec![
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
                vec![default.clone(), default.clone(), default.clone()],
            ],
        ])];
        let mut name_to_table_3d = HashMap::new();
        name_to_table_3d.insert(String::from("s3"), 0);

        let mut map = HashMap::new();
        let key = vec![0, 1, 0, 0];
        map.insert(key, set);
        let key = vec![0, 1, 0, 1];
        map.insert(key, default.clone());
        let key = vec![0, 1, 2, 0];
        map.insert(key, default.clone());
        let key = vec![0, 1, 2, 1];
        map.insert(key, default.clone());
        let tables = vec![table::Table::new(map, default)];
        let mut name_to_table = HashMap::new();
        name_to_table.insert(String::from("s4"), 0);

        let set_tables = TableData {
            name_to_constant,
            tables_1d,
            name_to_table_1d,
            tables_2d,
            name_to_table_2d,
            tables_3d,
            name_to_table_3d,
            tables,
            name_to_table,
        };

        let mut name_to_constant = HashMap::new();
        let vector = vec![0, 2];
        let default = Vec::new();
        name_to_constant.insert(String::from("v0"), vector.clone());

        let tables_1d = vec![table::Table1D::new(vec![
            vector.clone(),
            default.clone(),
            default.clone(),
        ])];
        let mut name_to_table_1d = HashMap::new();
        name_to_table_1d.insert(String::from("v1"), 0);

        let tables_2d = vec![table::Table2D::new(vec![
            vec![vector.clone(), default.clone(), default.clone()],
            vec![default.clone(), default.clone(), default.clone()],
            vec![default.clone(), default.clone(), default.clone()],
        ])];
        let mut name_to_table_2d = HashMap::new();
        name_to_table_2d.insert(String::from("v2"), 0);

        let tables_3d = vec![table::Table3D::new(vec![
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
                vec![default.clone(), default.clone(), default.clone()],
            ],
        ])];
        let mut name_to_table_3d = HashMap::new();
        name_to_table_3d.insert(String::from("v3"), 0);

        let mut map = HashMap::new();
        let key = vec![0, 1, 0, 0];
        map.insert(key, vector);
        let key = vec![0, 1, 0, 1];
        map.insert(key, default.clone());
        let key = vec![0, 1, 2, 0];
        map.insert(key, default.clone());
        let key = vec![0, 1, 2, 1];
        map.insert(key, default.clone());
        let tables = vec![table::Table::new(map, default)];
        let mut name_to_table = HashMap::new();
        name_to_table.insert(String::from("v4"), 0);

        let vector_tables = TableData {
            name_to_constant,
            tables_1d,
            name_to_table_1d,
            tables_2d,
            name_to_table_2d,
            tables_3d,
            name_to_table_3d,
            tables,
            name_to_table,
        };

        TableRegistry {
            integer_tables,
            continuous_tables,
            set_tables,
            vector_tables,
            bool_tables,
        }
    }

    #[test]
    fn table_registry_get_name_set() {
        let registry = generate_registry();
        let mut expected = collections::HashSet::new();
        expected.insert(String::from("i0"));
        expected.insert(String::from("i1"));
        expected.insert(String::from("i2"));
        expected.insert(String::from("i3"));
        expected.insert(String::from("i4"));
        expected.insert(String::from("c0"));
        expected.insert(String::from("c1"));
        expected.insert(String::from("c2"));
        expected.insert(String::from("c3"));
        expected.insert(String::from("c4"));
        expected.insert(String::from("b0"));
        expected.insert(String::from("b1"));
        expected.insert(String::from("b2"));
        expected.insert(String::from("b3"));
        expected.insert(String::from("b4"));
        expected.insert(String::from("s0"));
        expected.insert(String::from("s1"));
        expected.insert(String::from("s2"));
        expected.insert(String::from("s3"));
        expected.insert(String::from("s4"));
        expected.insert(String::from("v0"));
        expected.insert(String::from("v1"));
        expected.insert(String::from("v2"));
        expected.insert(String::from("v3"));
        expected.insert(String::from("v4"));
        assert_eq!(registry.get_name_set(), expected);
    }

    #[test]
    fn load_from_yaml_ok() {
        let metadata = generate_metadata();
        let expected = generate_registry();

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

        let registry = TableRegistry::load_from_yaml(tables, table_values, &metadata);
        println!("{:?}", registry);
        assert!(registry.is_ok());
        let registry = registry.unwrap();
        assert_eq!(registry.integer_tables, expected.integer_tables);
        assert_relative_eq!(registry.continuous_tables, expected.continuous_tables);
        assert_eq!(registry.bool_tables, expected.bool_tables);
    }

    #[test]
    fn load_from_yaml_err() {
        let metadata = generate_metadata();

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

        let registry = TableRegistry::load_from_yaml(tables, table_values, &metadata);
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

        let registry = TableRegistry::load_from_yaml(tables, table_values, &metadata);
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

        let registry = TableRegistry::load_from_yaml(tables, table_values, &metadata);
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

        let registry = TableRegistry::load_from_yaml(tables, table_values, &metadata);
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
        let registry = TableRegistry::load_from_yaml(tables, table_values, &metadata);
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
        let registry = TableRegistry::load_from_yaml(tables, table_values, &metadata);
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
        let registry = TableRegistry::load_from_yaml(tables, table_values, &metadata);
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
        let registry = TableRegistry::load_from_yaml(tables, table_values, &metadata);
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
        let registry = TableRegistry::load_from_yaml(tables, table_values, &metadata);
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
        let registry = TableRegistry::load_from_yaml(tables, table_values, &metadata);
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
        let registry = TableRegistry::load_from_yaml(tables, table_values, &metadata);
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
        let registry = TableRegistry::load_from_yaml(tables, table_values, &metadata);
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
        let registry = TableRegistry::load_from_yaml(tables, table_values, &metadata);
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
        let registry = TableRegistry::load_from_yaml(tables, table_values, &metadata);
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
        let registry = TableRegistry::load_from_yaml(tables, table_values, &metadata);
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
        let registry = TableRegistry::load_from_yaml(tables, table_values, &metadata);
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
        let registry = TableRegistry::load_from_yaml(tables, table_values, &metadata);
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
        let registry = TableRegistry::load_from_yaml(tables, table_values, &metadata);
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
        let registry = TableRegistry::load_from_yaml(tables, table_values, &metadata);
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
        let registry = TableRegistry::load_from_yaml(tables, table_values, &metadata);
        assert!(registry.is_err());
    }
}
