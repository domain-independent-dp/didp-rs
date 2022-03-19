use crate::state;
use crate::table;
use crate::variable;
use crate::yaml_util;
use approx::{AbsDiffEq, RelativeEq};
use std::collections;
use std::fmt;
use std::str;
use yaml_rust::Yaml;

#[derive(Debug, PartialEq, Clone, Default)]
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

impl<T: Copy + AbsDiffEq> AbsDiffEq for TableData<T>
where
    T::Epsilon: Copy,
{
    type Epsilon = T::Epsilon;

    fn default_epsilon() -> T::Epsilon {
        T::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: T::Epsilon) -> bool {
        self.name_to_table_1d == other.name_to_table_1d
            && self.name_to_table_2d == other.name_to_table_2d
            && self.name_to_table_3d == other.name_to_table_3d
            && self.name_to_table == other.name_to_table
            && self
                .tables_1d
                .iter()
                .zip(other.tables_1d.iter())
                .all(|(x, y)| x.abs_diff_eq(y, epsilon))
            && self
                .tables_2d
                .iter()
                .zip(other.tables_2d.iter())
                .all(|(x, y)| x.abs_diff_eq(y, epsilon))
            && self
                .tables_3d
                .iter()
                .zip(other.tables_3d.iter())
                .all(|(x, y)| x.abs_diff_eq(y, epsilon))
            && self
                .tables
                .iter()
                .zip(other.tables.iter())
                .all(|(x, y)| x.abs_diff_eq(y, epsilon))
    }
}

impl<T: Copy + RelativeEq> RelativeEq for TableData<T>
where
    T::Epsilon: Copy,
{
    fn default_max_relative() -> T::Epsilon {
        T::default_max_relative()
    }

    fn relative_eq(&self, other: &Self, epsilon: T::Epsilon, max_relative: T::Epsilon) -> bool {
        self.name_to_table_1d == other.name_to_table_1d
            && self.name_to_table_2d == other.name_to_table_2d
            && self.name_to_table_3d == other.name_to_table_3d
            && self.name_to_table == other.name_to_table
            && self
                .tables_1d
                .iter()
                .zip(other.tables_1d.iter())
                .all(|(x, y)| x.relative_eq(y, epsilon, max_relative))
            && self
                .tables_2d
                .iter()
                .zip(other.tables_2d.iter())
                .all(|(x, y)| x.relative_eq(y, epsilon, max_relative))
            && self
                .tables_3d
                .iter()
                .zip(other.tables_3d.iter())
                .all(|(x, y)| x.relative_eq(y, epsilon, max_relative))
            && self
                .tables
                .iter()
                .zip(other.tables.iter())
                .all(|(x, y)| x.relative_eq(y, epsilon, max_relative))
    }
}

#[derive(Debug, PartialEq, Clone, Default)]
pub struct TableRegistry {
    pub integer_tables: TableData<variable::Integer>,
    pub continuous_tables: TableData<variable::Continuous>,
    pub bool_tables: TableData<bool>,
}

enum TableReturnType {
    Integer(variable::Integer),
    Continuous(variable::Continuous),
    Bool(bool),
}

impl TableRegistry {
    pub fn load_from_yaml(
        tables: &Yaml,
        table_values: &Yaml,
        metadata: &state::StateMetadata,
    ) -> Result<TableRegistry, yaml_util::YamlContentErr> {
        let tables = yaml_util::get_array(tables)?;
        let mut table_names = Vec::with_capacity(tables.len());
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
        let mut integer_tables_1d = Vec::new();
        let mut integer_name_to_table_1d = collections::HashMap::new();
        let mut integer_tables_2d = Vec::new();
        let mut integer_name_to_table_2d = collections::HashMap::new();
        let mut integer_tables_3d = Vec::new();
        let mut integer_name_to_table_3d = collections::HashMap::new();
        let mut integer_tables = Vec::new();
        let mut integer_name_to_table = collections::HashMap::new();
        let mut continuous_tables_1d = Vec::new();
        let mut continuous_name_to_table_1d = collections::HashMap::new();
        let mut continuous_tables_2d = Vec::new();
        let mut continuous_name_to_table_2d = collections::HashMap::new();
        let mut continuous_tables_3d = Vec::new();
        let mut continuous_name_to_table_3d = collections::HashMap::new();
        let mut continuous_tables = Vec::new();
        let mut continuous_name_to_table = collections::HashMap::new();
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
            if arg_types.len() == 1 {
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
                }
            }
        }
        Ok(TableRegistry {
            integer_tables: TableData {
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
                tables_1d: continuous_tables_1d,
                name_to_table_1d: continuous_name_to_table_1d,
                tables_2d: continuous_tables_2d,
                name_to_table_2d: continuous_name_to_table_2d,
                tables_3d: continuous_tables_3d,
                name_to_table_3d: continuous_name_to_table_3d,
                tables: continuous_tables,
                name_to_table: continuous_name_to_table,
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
                    "`{}` is greater than the number of the object for table",
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
            tables_1d,
            name_to_table_1d,
            tables_2d,
            name_to_table_2d,
            tables_3d,
            name_to_table_3d,
            tables,
            name_to_table,
        };

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
        name_to_table.insert(String::from("i4"), 0);

        let continuous_tables = TableData {
            tables_1d,
            name_to_table_1d,
            tables_2d,
            name_to_table_2d,
            tables_3d,
            name_to_table_3d,
            tables,
            name_to_table,
        };

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
            bool_tables,
        }
    }

    #[test]
    fn load_from_yaml_ok() {
        let metadata = generate_metadata();
        let expected = generate_registry();

        let tables = r"
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
";
        let table_values = r"
i1:
      0: 10
      1: 20
      2: 30
i2: { [0, 0]: 10, [0, 1]: 20, [0, 2]: 30 }
i3: { [0, 0, 0]: 10, [0, 0, 1]: 20, [0, 0, 2]: 30 }
i4: { [0, 1, 0, 0]: 100, [0, 1, 0, 1]: 200, [0, 1, 2, 0]: 300, [0, 1, 2, 1]: 400 }
c1:
      0: 10
      1: 20
      2: 30
c2: { [0, 0]: 10, [0, 1]: 20, [0, 2]: 30 }
c3: { [0, 0, 0]: 10, [0, 0, 1]: 20, [0, 0, 2]: 30 }
c4: { [0, 1, 0, 0]: 100, [0, 1, 0, 1]: 200, [0, 1, 2, 0]: 300, [0, 1, 2, 1]: 400 }
b1: { 0: true, 1: false, 2: false }
b2: { [0, 0]: true }
b3: { [0, 0, 0]: true, [1, 0, 0]: true, [2, 0, 0]: true }
b4: { [0, 1, 0, 0]: true, [0, 1, 0, 1]: false, [0, 1, 2, 0]: false, [0, 1, 2, 1]: false }
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
- name: f1
  type: integer
  object: null
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
    }
}
