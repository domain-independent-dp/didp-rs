use crate::state;
use crate::table;
use crate::variable;
use crate::yaml_util;
use std::collections;
use std::fmt;
use std::str;
use yaml_rust::Yaml;

pub struct TableRegistry<T: variable::Numeric> {
    pub tables_1d: Vec<table::Table1D<T>>,
    pub name_to_table_1d: collections::HashMap<String, usize>,
    pub tables_2d: Vec<table::Table2D<T>>,
    pub name_to_table_2d: collections::HashMap<String, usize>,
    pub tables_3d: Vec<table::Table3D<T>>,
    pub name_to_table_3d: collections::HashMap<String, usize>,
    pub tables: Vec<table::Table<T>>,
    pub name_to_table: collections::HashMap<String, usize>,
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
        let mut name_to_arg_types = collections::HashMap::new();
        let mut name_to_default_value = collections::HashMap::new();
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
            name_to_arg_types.insert(name.clone(), arg_types);
            if let Ok(value) = yaml_util::get_numeric_by_key(map, "default") {
                name_to_default_value.insert(name.clone(), value);
            } else {
                name_to_default_value.insert(name.clone(), T::zero());
            }
        }
        let mut tables_1d = Vec::new();
        let mut name_to_table_1d = collections::HashMap::new();
        let mut tables_2d = Vec::new();
        let mut name_to_table_2d = collections::HashMap::new();
        let mut tables_3d = Vec::new();
        let mut name_to_table_3d = collections::HashMap::new();
        let mut tables = Vec::new();
        let mut name_to_table = collections::HashMap::new();
        let table_values = yaml_util::get_map(table_values)?;
        for (name, args_types) in name_to_arg_types {
            let value = yaml_util::get_yaml_by_key(table_values, &name)?;
            let default = *name_to_default_value.get(&name).unwrap();
            if args_types.len() == 1 {
                let size = metadata.object_numbers[args_types[0]];
                let f = Self::load_table_1d_from_yaml(value, size, default)?;
                name_to_table_1d.insert(name, tables_1d.len());
                tables_1d.push(f);
            } else if args_types.len() == 2 {
                let size_x = metadata.object_numbers[args_types[0]];
                let size_y = metadata.object_numbers[args_types[1]];
                let f = Self::load_table_2d_from_yaml(value, size_x, size_y, default)?;
                name_to_table_2d.insert(name, tables_2d.len());
                tables_2d.push(f);
            } else if args_types.len() == 3 {
                let size_x = metadata.object_numbers[args_types[0]];
                let size_y = metadata.object_numbers[args_types[1]];
                let size_z = metadata.object_numbers[args_types[2]];
                let f = Self::load_table_3d_from_yaml(value, size_x, size_y, size_z, default)?;
                name_to_table_3d.insert(name, tables_3d.len());
                tables_3d.push(f);
            } else {
                let size: Vec<usize> = args_types
                    .iter()
                    .map(|i| metadata.object_numbers[*i])
                    .collect();
                let f = Self::load_table_from_yaml(value, size, default)?;
                name_to_table.insert(name, tables.len());
                tables.push(f);
            }
        }
        Ok(TableRegistry {
            tables_1d,
            name_to_table_1d,
            tables_2d,
            name_to_table_2d,
            tables_3d,
            name_to_table_3d,
            tables,
            name_to_table,
        })
    }

    fn load_table_1d_from_yaml(
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

    fn load_table_2d_from_yaml(
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

    fn load_table_3d_from_yaml(
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

    fn load_table_from_yaml(
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
}
