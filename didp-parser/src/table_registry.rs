use crate::state;
use crate::table;
use crate::variable;
use crate::yaml_util;
use std::collections;
use std::fmt;
use std::str;
use yaml_rust::Yaml;

pub struct TableData<T: Copy> {
    pub tables_1d: Vec<table::Table1D<T>>,
    pub name_to_table_1d: collections::HashMap<String, usize>,
    pub tables_2d: Vec<table::Table2D<T>>,
    pub name_to_table_2d: collections::HashMap<String, usize>,
    pub tables_3d: Vec<table::Table3D<T>>,
    pub name_to_table_3d: collections::HashMap<String, usize>,
    pub tables: Vec<table::Table<T>>,
    pub name_to_table: collections::HashMap<String, usize>,
}

pub struct TableRegistry<T: variable::Numeric> {
    pub numeric_tables: TableData<T>,
    pub bool_tables: TableData<bool>,
}

enum TableReturnType<T: variable::Numeric> {
    Bool(bool),
    Numeric(T),
}

impl<T: variable::Numeric> TableRegistry<T> {
    pub fn load_from_yaml(
        tables: &Yaml,
        table_values: &Yaml,
        metadata: &state::StateMetadata,
    ) -> Result<TableRegistry<T>, yaml_util::YamlContentErr>
    where
        <T as str::FromStr>::Err: fmt::Debug,
    {
        let tables = yaml_util::get_array(tables)?;
        let mut name_to_signature = collections::HashMap::new();
        for value in tables {
            let map = yaml_util::get_map(value)?;
            let name = yaml_util::get_string_by_key(map, "name")?;
            let args = yaml_util::get_string_array_by_key(map, "args")?;
            if args.is_empty() {
                return Err(yaml_util::YamlContentErr::new(
                    "table has no arguments".to_string(),
                ));
            }
            let mut arg_types = Vec::with_capacity(args.len());
            for object in &args {
                if let Some(value) = metadata.name_to_element_variable.get(object) {
                    arg_types.push(*value);
                } else {
                    return Err(yaml_util::YamlContentErr::new(format!(
                        "object `{}` does not exist",
                        object
                    )));
                }
            }
            let return_type = yaml_util::get_string_by_key(map, "type")?;
            match &return_type[..] {
                "numeric" => {
                    if let Ok(value) = yaml_util::get_numeric_by_key(map, "default") {
                        name_to_signature
                            .insert(name.clone(), (arg_types, TableReturnType::Numeric(value)));
                    } else {
                        name_to_signature.insert(
                            name.clone(),
                            (arg_types, TableReturnType::Numeric(T::zero())),
                        );
                    }
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
        }
        let mut numeric_tables_1d = Vec::new();
        let mut numeric_name_to_table_1d = collections::HashMap::new();
        let mut numeric_tables_2d = Vec::new();
        let mut numeric_name_to_table_2d = collections::HashMap::new();
        let mut numeric_tables_3d = Vec::new();
        let mut numeric_name_to_table_3d = collections::HashMap::new();
        let mut numeric_tables = Vec::new();
        let mut numeric_name_to_table = collections::HashMap::new();
        let mut bool_tables_1d = Vec::new();
        let mut bool_name_to_table_1d = collections::HashMap::new();
        let mut bool_tables_2d = Vec::new();
        let mut bool_name_to_table_2d = collections::HashMap::new();
        let mut bool_tables_3d = Vec::new();
        let mut bool_name_to_table_3d = collections::HashMap::new();
        let mut bool_tables = Vec::new();
        let mut bool_name_to_table = collections::HashMap::new();
        let table_values = yaml_util::get_map(table_values)?;
        for (name, signature) in name_to_signature {
            let arg_types = signature.0;
            let value = yaml_util::get_yaml_by_key(table_values, &name)?;
            if arg_types.len() == 1 {
                let size = metadata.object_numbers[arg_types[0]];
                match signature.1 {
                    TableReturnType::Numeric(default) => {
                        let f = Self::load_numeric_table_1d_from_yaml(value, size, default)?;
                        numeric_name_to_table_1d.insert(name, numeric_tables_1d.len());
                        numeric_tables_1d.push(f);
                    }
                    TableReturnType::Bool(default) => {
                        let f = Self::load_bool_table_1d_from_yaml(value, size, default)?;
                        bool_name_to_table_1d.insert(name, bool_tables_1d.len());
                        bool_tables_1d.push(f);
                    }
                }
            } else if arg_types.len() == 2 {
                let size_x = metadata.object_numbers[arg_types[0]];
                let size_y = metadata.object_numbers[arg_types[1]];
                match signature.1 {
                    TableReturnType::Numeric(default) => {
                        let f =
                            Self::load_numeric_table_2d_from_yaml(value, size_x, size_y, default)?;
                        numeric_name_to_table_2d.insert(name, numeric_tables_2d.len());
                        numeric_tables_2d.push(f);
                    }
                    TableReturnType::Bool(default) => {
                        let f = Self::load_bool_table_2d_from_yaml(value, size_x, size_y, default)?;
                        bool_name_to_table_2d.insert(name, bool_tables_2d.len());
                        bool_tables_2d.push(f);
                    }
                }
            } else if arg_types.len() == 3 {
                let size_x = metadata.object_numbers[arg_types[0]];
                let size_y = metadata.object_numbers[arg_types[1]];
                let size_z = metadata.object_numbers[arg_types[2]];
                match signature.1 {
                    TableReturnType::Numeric(default) => {
                        let f = Self::load_numeric_table_3d_from_yaml(
                            value, size_x, size_y, size_z, default,
                        )?;
                        numeric_name_to_table_3d.insert(name, numeric_tables_3d.len());
                        numeric_tables_3d.push(f);
                    }
                    TableReturnType::Bool(default) => {
                        let f = Self::load_bool_table_3d_from_yaml(
                            value, size_x, size_y, size_z, default,
                        )?;
                        bool_name_to_table_3d.insert(name, bool_tables_3d.len());
                        bool_tables_3d.push(f);
                    }
                }
            } else {
                let size: Vec<usize> = arg_types
                    .iter()
                    .map(|i| metadata.object_numbers[*i])
                    .collect();
                match signature.1 {
                    TableReturnType::Numeric(default) => {
                        let f = Self::load_numeric_table_from_yaml(value, size, default)?;
                        numeric_name_to_table.insert(name, tables.len());
                        numeric_tables.push(f);
                    }
                    TableReturnType::Bool(default) => {
                        let f = Self::load_bool_table_from_yaml(value, size, default)?;
                        bool_name_to_table.insert(name, tables.len());
                        bool_tables.push(f);
                    }
                }
            }
        }
        Ok(TableRegistry {
            numeric_tables: TableData {
                tables_1d: numeric_tables_1d,
                name_to_table_1d: numeric_name_to_table_1d,
                tables_2d: numeric_tables_2d,
                name_to_table_2d: numeric_name_to_table_2d,
                tables_3d: numeric_tables_3d,
                name_to_table_3d: numeric_name_to_table_3d,
                tables: numeric_tables,
                name_to_table: numeric_name_to_table,
            },
            bool_tables: TableData {
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

    fn load_numeric_table_1d_from_yaml(
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
                    "`{}` is greater than the number of the object for table",
                    args,
                )));
            }
            body[args] = value;
        }
        Ok(table::Table1D::new(body))
    }

    fn load_numeric_table_2d_from_yaml(
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

    fn load_numeric_table_3d_from_yaml(
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

    fn load_numeric_table_from_yaml(
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
                    "`{}` is greater than the number of the object for table",
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
    ) -> Result<table::Table<bool>, yaml_util::YamlContentErr>
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
}
