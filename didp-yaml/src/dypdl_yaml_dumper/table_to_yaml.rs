use itertools::Itertools;
use std::error::Error;

use super::ToYaml;
use dypdl::{HasShape, Table, Table1D, Table2D, Table3D};
use dypdl::{ModelErr, Set, TableData};
use rustc_hash::FxHashMap;
use std::num::TryFromIntError;
use yaml_rust::{
    yaml::{Array, Hash},
    Yaml,
};

impl<T: ToYaml> ToYaml for Table1D<T> {
    fn to_yaml(&self) -> Result<Yaml, Box<dyn Error>> {
        let mut hash = Hash::new();
        let x_range = 0..self.0.len();
        for i in x_range {
            hash.insert(i.to_yaml()?, self.get(i).to_yaml()?);
        }
        Ok(Yaml::Hash(hash))
    }
}

impl<T: ToYaml> ToYaml for Table2D<T> {
    fn to_yaml(&self) -> Result<Yaml, Box<dyn Error>> {
        let mut hash = Hash::new();
        if self.0.is_empty() || self.0[0].is_empty() {
            return Ok(Yaml::Hash(hash));
        }

        let x_range = 0..self.0.len();
        let y_range = 0..self.0[0].len();
        for (i, j) in x_range.cartesian_product(y_range) {
            hash.insert(
                Yaml::Array(Array::from([i.to_yaml()?, j.to_yaml()?])),
                self.get(i, j).to_yaml()?,
            );
        }
        Ok(Yaml::Hash(hash))
    }
}

impl<T: ToYaml> ToYaml for Table3D<T> {
    fn to_yaml(&self) -> Result<Yaml, Box<dyn Error>> {
        let mut hash = Hash::new();
        if self.0.is_empty() || self.0[0].is_empty() || self.0[0][0].is_empty() {
            return Ok(Yaml::Hash(hash));
        }

        let x_range = 0..self.0.len();
        let y_range = 0..self.0[0].len();
        let z_range = 0..self.0[0][0].len();
        for ((i, j), k) in x_range
            .cartesian_product(y_range)
            .cartesian_product(z_range)
        {
            hash.insert(
                Yaml::Array(Array::from([i.to_yaml()?, j.to_yaml()?, k.to_yaml()?])),
                self.get(i, j, k).to_yaml()?,
            );
        }
        Ok(Yaml::Hash(hash))
    }
}

impl<T: ToYaml> ToYaml for Table<T> {
    fn to_yaml(&self) -> Result<yaml_rust::Yaml, Box<dyn Error>> {
        let mut hash = Hash::new();
        let mut keys = self.map.keys().collect::<Vec<_>>();
        keys.sort();

        for key in keys {
            let key_array: Result<Vec<Yaml>, Box<dyn Error>> =
                key.iter().map(|i: &usize| (*i).to_yaml()).collect();
            hash.insert(Yaml::Array(key_array?), self.get(key).to_yaml()?);
        }

        Ok(Yaml::Hash(hash))
    }
}

fn add_dictionary_to_yaml<V: ToYaml>(
    key: &String,
    table_type: &str,
    name_to_table: &FxHashMap<String, usize>,
    tables: &[Table<V>],
    dictionary_names: &mut Array,
    dictionary_values: &mut Hash,
) -> Result<(), Box<dyn Error>> {
    let mut dictionary_name_yaml = Hash::new();
    dictionary_name_yaml.insert(Yaml::from_str("name"), Yaml::from_str(key));
    dictionary_name_yaml.insert(Yaml::from_str("type"), Yaml::from_str(table_type));

    let table_index = name_to_table.get(key).unwrap();

    dictionary_name_yaml.insert(
        Yaml::from_str("default"),
        tables[*table_index].default.to_yaml()?,
    );
    dictionary_names.push(Yaml::Hash(dictionary_name_yaml));

    dictionary_values.insert(Yaml::from_str(key), tables[*table_index].to_yaml()?);
    Ok(())
}

fn add_table_nd_to_yaml<TableND: ToYaml + HasShape>(
    key: &String,
    table_type: &str,
    name_to_table: &FxHashMap<String, usize>,
    tables: &[TableND],
    table_names: &mut Array,
    table_values: &mut Hash,
) -> Result<(), Box<dyn Error>> {
    let table_index = name_to_table.get(key).unwrap();
    let table = &tables[*table_index];
    let table_sizes = table.shape();

    let mut table_name_yaml = Hash::new();
    table_name_yaml.insert(Yaml::from_str("name"), Yaml::from_str(key));
    table_name_yaml.insert(Yaml::from_str("type"), Yaml::from_str(table_type));
    table_name_yaml.insert(
        Yaml::from_str("args"),
        Yaml::Array({
            let array: Result<Vec<Yaml>, TryFromIntError> = table_sizes
                .iter()
                .map(|size| -> Result<Yaml, TryFromIntError> {
                    Ok(Yaml::Integer(i64::try_from(*size)?))
                })
                .collect();
            array?
        }),
    );
    table_names.push(Yaml::Hash(table_name_yaml));

    table_values.insert(Yaml::from_str(key), table.to_yaml()?);
    Ok(())
}

pub fn table_data_to_yaml<T: ToYaml>(
    table_data: &TableData<T>,
    table_type: &str,
    table_names: &mut Array,
    table_values: &mut Hash,
    dictionary_names: &mut Array,
    dictionary_values: &mut Hash,
) -> Result<(), Box<dyn Error>> {
    for key in table_data.name_to_constant.keys() {
        let mut table_name_yaml = Hash::new();
        table_name_yaml.insert(Yaml::from_str("name"), Yaml::from_str(key));
        table_name_yaml.insert(Yaml::from_str("type"), Yaml::from_str(table_type));

        table_names.push(Yaml::Hash(table_name_yaml));

        table_values.insert(
            Yaml::from_str(key),
            table_data.name_to_constant.get(key).unwrap().to_yaml()?,
        );
    }

    // Output the tables in the same order as they are stored in the table registry.
    // The value in the name_to_table map is the index of the table in the vector of table data.
    let mut key_value_pairs: Vec<(String, usize)> =
        table_data.name_to_table_1d.clone().drain().collect();
    key_value_pairs.sort_by_key(|k| k.1);
    for (key, _) in key_value_pairs {
        add_table_nd_to_yaml(
            &key,
            table_type,
            &table_data.name_to_table_1d,
            &table_data.tables_1d,
            table_names,
            table_values,
        )?;
    }

    let mut key_value_pairs: Vec<(String, usize)> =
        table_data.name_to_table_2d.clone().drain().collect();
    key_value_pairs.sort_by_key(|k| k.1);
    for (key, _) in key_value_pairs {
        add_table_nd_to_yaml(
            &key,
            table_type,
            &table_data.name_to_table_2d,
            &table_data.tables_2d,
            table_names,
            table_values,
        )?;
    }

    let mut key_value_pairs: Vec<(String, usize)> =
        table_data.name_to_table_3d.clone().drain().collect();
    key_value_pairs.sort_by_key(|k| k.1);
    for (key, _) in key_value_pairs {
        add_table_nd_to_yaml(
            &key,
            table_type,
            &table_data.name_to_table_3d,
            &table_data.tables_3d,
            table_names,
            table_values,
        )?;
    }

    for key in table_data.name_to_table.keys() {
        add_dictionary_to_yaml(
            key,
            table_type,
            &table_data.name_to_table,
            &table_data.tables,
            dictionary_names,
            dictionary_values,
        )?;
    }

    Ok(())
}

// All set dictionaries and tables require another function to add them into the yaml, since
// they require 'object' field.
fn add_set_dictionary_to_yaml(
    key: &String,
    table_type: &str,
    set_size: usize,
    name_to_table: &FxHashMap<String, usize>,
    tables: &[Table<Set>],
    dictionary_names: &mut Array,
    dictionary_values: &mut Hash,
) -> Result<(), Box<dyn Error>> {
    let mut dictionary_name_yaml = Hash::new();
    dictionary_name_yaml.insert(Yaml::from_str("name"), Yaml::from_str(key));
    dictionary_name_yaml.insert(Yaml::from_str("type"), Yaml::from_str(table_type));
    dictionary_name_yaml.insert(
        Yaml::from_str("object"),
        Yaml::Integer(i64::try_from(set_size)?),
    );

    let table_index = name_to_table.get(key).unwrap();

    dictionary_name_yaml.insert(
        Yaml::from_str("default"),
        tables[*table_index].default.to_yaml()?,
    );
    dictionary_names.push(Yaml::Hash(dictionary_name_yaml));

    dictionary_values.insert(Yaml::from_str(key), tables[*table_index].to_yaml()?);
    Ok(())
}

fn add_set_table_nd_to_yaml<TableND: ToYaml + HasShape>(
    key: &String,
    table_type: &str,
    set_size: usize,
    name_to_table: &FxHashMap<String, usize>,
    tables: &[TableND],
    table_names: &mut Array,
    table_values: &mut Hash,
) -> Result<(), Box<dyn Error>> {
    let table_index = name_to_table.get(key).unwrap();
    let table = &tables[*table_index];
    let table_sizes = table.shape();

    let mut table_name_yaml = Hash::new();
    table_name_yaml.insert(Yaml::from_str("name"), Yaml::from_str(key));
    table_name_yaml.insert(Yaml::from_str("type"), Yaml::from_str(table_type));
    table_name_yaml.insert(
        Yaml::from_str("args"),
        Yaml::Array({
            let array: Result<Vec<Yaml>, TryFromIntError> = table_sizes
                .iter()
                .map(|size| -> Result<Yaml, TryFromIntError> {
                    Ok(Yaml::Integer(i64::try_from(*size)?))
                })
                .collect();
            array?
        }),
    );
    table_name_yaml.insert(
        Yaml::from_str("object"),
        Yaml::Integer(i64::try_from(set_size)?),
    );
    table_names.push(Yaml::Hash(table_name_yaml));

    table_values.insert(Yaml::from_str(key), table.to_yaml()?);
    Ok(())
}

pub fn set_table_data_to_yaml(
    table_data: &TableData<Set>,
    table_type: &str,
    table_names: &mut Array,
    table_values: &mut Hash,
    dictionary_names: &mut Array,
    dictionary_values: &mut Hash,
) -> Result<(), Box<dyn Error>> {
    for key in table_data.name_to_constant.keys() {
        let mut table_name_yaml = Hash::new();
        table_name_yaml.insert(Yaml::from_str("name"), Yaml::from_str(key));
        table_name_yaml.insert(Yaml::from_str("type"), Yaml::from_str(table_type));
        let set_size = table_data.name_to_constant.get(key).unwrap().len();
        table_name_yaml.insert(
            Yaml::from_str("object"),
            Yaml::Integer(i64::try_from(set_size)?),
        );

        table_names.push(Yaml::Hash(table_name_yaml));

        table_values.insert(
            Yaml::from_str(key),
            table_data.name_to_constant.get(key).unwrap().to_yaml()?,
        );
    }

    let mut key_value_pairs: Vec<(String, usize)> =
        table_data.name_to_table_1d.clone().drain().collect();
    key_value_pairs.sort_by_key(|k| k.1);
    for (key, value) in key_value_pairs {
        let table1d = &table_data.tables_1d[value];

        if table1d.shape().contains(&0) {
            return Err(ModelErr::new(format!("Empty Set Table1D: {}", key)).into());
        }
        let set_size = table1d.capacity_of_set();
        add_set_table_nd_to_yaml(
            &key,
            table_type,
            set_size,
            &table_data.name_to_table_1d,
            &table_data.tables_1d,
            table_names,
            table_values,
        )?;
    }

    let mut key_value_pairs: Vec<(String, usize)> =
        table_data.name_to_table_2d.clone().drain().collect();
    key_value_pairs.sort_by_key(|k| k.1);
    for (key, value) in key_value_pairs {
        let table2d = &table_data.tables_2d[value];

        if table2d.shape().contains(&0) {
            return Err(ModelErr::new(format!("Empty Set Table2D: {}", key)).into());
        }
        let set_size = table2d.capacity_of_set();
        add_set_table_nd_to_yaml(
            &key,
            table_type,
            set_size,
            &table_data.name_to_table_2d,
            &table_data.tables_2d,
            table_names,
            table_values,
        )?;
    }

    let mut key_value_pairs: Vec<(String, usize)> =
        table_data.name_to_table_3d.clone().drain().collect();
    key_value_pairs.sort_by_key(|k| k.1);
    for (key, value) in key_value_pairs {
        let table3d = &table_data.tables_3d[value];

        if table3d.shape().contains(&0) {
            return Err(ModelErr::new(format!("Empty Set Table3D: {}", key)).into());
        }
        let set_size = table3d.capacity_of_set();
        add_set_table_nd_to_yaml(
            &key,
            table_type,
            set_size,
            &table_data.name_to_table_3d,
            &table_data.tables_3d,
            table_names,
            table_values,
        )?;
    }

    for key in table_data.name_to_table.keys() {
        let table_index = &table_data.name_to_table.get(key);
        let table = &table_data.tables[*table_index.unwrap()];
        let set_size = table.capacity_of_set();

        add_set_dictionary_to_yaml(
            key,
            table_type,
            set_size,
            &table_data.name_to_table,
            &table_data.tables,
            dictionary_names,
            dictionary_values,
        )?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use dypdl::{Set, Table, Table1D, Table2D, Table3D};
    use rustc_hash::FxHashMap;
    use yaml_rust::{
        yaml::{Array, Hash},
        Yaml,
    };

    use super::{set_table_data_to_yaml, table_data_to_yaml};

    #[test]
    fn integer_table_data_to_yaml() {
        let mut name_to_constant = FxHashMap::default();
        name_to_constant.insert(String::from("i0"), 0);

        let tables_1d = vec![Table1D::new(vec![0])];
        let mut name_to_table_1d = FxHashMap::default();
        name_to_table_1d.insert(String::from("i1"), 0);

        let tables_2d = vec![Table2D::new(vec![vec![0]])];
        let mut name_to_table_2d = FxHashMap::default();
        name_to_table_2d.insert(String::from("i2"), 0);

        let tables_3d = vec![Table3D::new(vec![vec![vec![0]]])];
        let mut name_to_table_3d = FxHashMap::default();
        name_to_table_3d.insert(String::from("i3"), 0);

        let tables = vec![Table::new(FxHashMap::default(), 0)];
        let mut name_to_table = FxHashMap::default();
        name_to_table.insert(String::from("i4"), 0);

        let integer_tables = dypdl::TableData {
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

        let mut table_names = Array::new();
        let mut table_values = Hash::new();
        let mut dictionary_names = Array::new();
        let mut dictionary_values = Hash::new();
        let result = table_data_to_yaml(
            &integer_tables,
            "integer",
            &mut table_names,
            &mut table_values,
            &mut dictionary_names,
            &mut dictionary_values,
        );
        assert!(result.is_ok());

        let mut expected_table_names = Array::new();
        let mut i0 = Hash::new();
        i0.insert(Yaml::from_str("name"), Yaml::from_str("i0"));
        i0.insert(Yaml::from_str("type"), Yaml::from_str("integer"));
        expected_table_names.push(Yaml::Hash(i0));

        let mut i1 = Hash::new();
        i1.insert(Yaml::from_str("name"), Yaml::from_str("i1"));
        i1.insert(Yaml::from_str("type"), Yaml::from_str("integer"));
        i1.insert(Yaml::from_str("args"), Yaml::Array(vec![Yaml::Integer(1)]));
        expected_table_names.push(Yaml::Hash(i1));

        let mut i2 = Hash::new();
        i2.insert(Yaml::from_str("name"), Yaml::from_str("i2"));
        i2.insert(Yaml::from_str("type"), Yaml::from_str("integer"));
        i2.insert(
            Yaml::from_str("args"),
            Yaml::Array(vec![Yaml::Integer(1), Yaml::Integer(1)]),
        );
        expected_table_names.push(Yaml::Hash(i2));

        let mut i3 = Hash::new();
        i3.insert(Yaml::from_str("name"), Yaml::from_str("i3"));
        i3.insert(Yaml::from_str("type"), Yaml::from_str("integer"));
        i3.insert(
            Yaml::from_str("args"),
            Yaml::Array(vec![Yaml::Integer(1), Yaml::Integer(1), Yaml::Integer(1)]),
        );
        expected_table_names.push(Yaml::Hash(i3));

        assert_eq!(table_names, expected_table_names);

        let mut expected_dictionary_names = Array::new();
        let mut i4 = Hash::new();
        i4.insert(Yaml::from_str("name"), Yaml::from_str("i4"));
        i4.insert(Yaml::from_str("type"), Yaml::from_str("integer"));
        i4.insert(Yaml::from_str("default"), Yaml::Integer(0));
        expected_dictionary_names.push(Yaml::Hash(i4));

        assert_eq!(dictionary_names, expected_dictionary_names);

        let mut expected_table_values = Hash::new();
        expected_table_values.insert(Yaml::from_str("i0"), Yaml::Integer(0));
        let mut i1_values = Hash::new();
        i1_values.insert(Yaml::Integer(0), Yaml::Integer(0));
        expected_table_values.insert(Yaml::from_str("i1"), Yaml::Hash(i1_values));
        let mut i2_values = Hash::new();
        i2_values.insert(
            Yaml::Array(vec![Yaml::Integer(0), Yaml::Integer(0)]),
            Yaml::Integer(0),
        );
        expected_table_values.insert(Yaml::from_str("i2"), Yaml::Hash(i2_values));
        let mut i3_values = Hash::new();
        i3_values.insert(
            Yaml::Array(vec![Yaml::Integer(0), Yaml::Integer(0), Yaml::Integer(0)]),
            Yaml::Integer(0),
        );
        expected_table_values.insert(Yaml::from_str("i3"), Yaml::Hash(i3_values));

        assert_eq!(table_values, expected_table_values);

        let mut expected_dictionary_values = Hash::new();
        expected_dictionary_values.insert(Yaml::from_str("i4"), Yaml::Hash(Hash::new()));

        assert_eq!(dictionary_values, expected_dictionary_values);
    }

    #[test]
    fn continuous_table_data_to_yaml() {
        let mut name_to_constant = FxHashMap::default();
        name_to_constant.insert(String::from("c0"), 0.0);

        let tables_1d = vec![Table1D::new(vec![0.0])];
        let mut name_to_table_1d = FxHashMap::default();
        name_to_table_1d.insert(String::from("c1"), 0);

        let tables_2d = vec![Table2D::new(vec![vec![0.0]])];
        let mut name_to_table_2d = FxHashMap::default();
        name_to_table_2d.insert(String::from("c2"), 0);

        let tables_3d = vec![Table3D::new(vec![vec![vec![0.0]]])];
        let mut name_to_table_3d = FxHashMap::default();
        name_to_table_3d.insert(String::from("c3"), 0);

        let tables = vec![Table::new(FxHashMap::default(), 0.0)];
        let mut name_to_table = FxHashMap::default();
        name_to_table.insert(String::from("c4"), 0);

        let continuous_tables = dypdl::TableData {
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

        let mut table_names = Array::new();
        let mut table_values = Hash::new();
        let mut dictionary_names = Array::new();
        let mut dictionary_values = Hash::new();
        let result = table_data_to_yaml(
            &continuous_tables,
            "continuous",
            &mut table_names,
            &mut table_values,
            &mut dictionary_names,
            &mut dictionary_values,
        );
        assert!(result.is_ok());

        let mut expected_table_names = Array::new();
        let mut c0 = Hash::new();
        c0.insert(Yaml::from_str("name"), Yaml::from_str("c0"));
        c0.insert(Yaml::from_str("type"), Yaml::from_str("continuous"));
        expected_table_names.push(Yaml::Hash(c0));

        let mut c1 = Hash::new();
        c1.insert(Yaml::from_str("name"), Yaml::from_str("c1"));
        c1.insert(Yaml::from_str("type"), Yaml::from_str("continuous"));
        c1.insert(Yaml::from_str("args"), Yaml::Array(vec![Yaml::Integer(1)]));
        expected_table_names.push(Yaml::Hash(c1));

        let mut c2 = Hash::new();
        c2.insert(Yaml::from_str("name"), Yaml::from_str("c2"));
        c2.insert(Yaml::from_str("type"), Yaml::from_str("continuous"));
        c2.insert(
            Yaml::from_str("args"),
            Yaml::Array(vec![Yaml::Integer(1), Yaml::Integer(1)]),
        );
        expected_table_names.push(Yaml::Hash(c2));

        let mut c3 = Hash::new();
        c3.insert(Yaml::from_str("name"), Yaml::from_str("c3"));
        c3.insert(Yaml::from_str("type"), Yaml::from_str("continuous"));
        c3.insert(
            Yaml::from_str("args"),
            Yaml::Array(vec![Yaml::Integer(1), Yaml::Integer(1), Yaml::Integer(1)]),
        );
        expected_table_names.push(Yaml::Hash(c3));

        assert_eq!(table_names, expected_table_names);

        let mut expected_dictionary_names = Array::new();
        let mut c4 = Hash::new();
        c4.insert(Yaml::from_str("name"), Yaml::from_str("c4"));
        c4.insert(Yaml::from_str("type"), Yaml::from_str("continuous"));
        c4.insert(Yaml::from_str("default"), Yaml::Real("0".to_owned()));
        expected_dictionary_names.push(Yaml::Hash(c4));

        assert_eq!(dictionary_names, expected_dictionary_names);

        let mut expected_table_values = Hash::new();
        expected_table_values.insert(Yaml::from_str("c0"), Yaml::Real("0".to_owned()));
        let mut c1_values = Hash::new();
        c1_values.insert(Yaml::Integer(0), Yaml::Real("0".to_owned()));
        expected_table_values.insert(Yaml::from_str("c1"), Yaml::Hash(c1_values));
        let mut c2_values = Hash::new();
        c2_values.insert(
            Yaml::Array(vec![Yaml::Integer(0), Yaml::Integer(0)]),
            Yaml::Real("0".to_owned()),
        );
        expected_table_values.insert(Yaml::from_str("c2"), Yaml::Hash(c2_values));
        let mut c3_values = Hash::new();
        c3_values.insert(
            Yaml::Array(vec![Yaml::Integer(0), Yaml::Integer(0), Yaml::Integer(0)]),
            Yaml::Real("0".to_owned()),
        );
        expected_table_values.insert(Yaml::from_str("c3"), Yaml::Hash(c3_values));

        assert_eq!(table_values, expected_table_values);

        let mut expected_dictionary_values = Hash::new();
        expected_dictionary_values.insert(Yaml::from_str("c4"), Yaml::Hash(Hash::new()));

        assert_eq!(dictionary_values, expected_dictionary_values);
    }

    #[test]
    fn boolean_table_data_to_yaml() {
        let mut name_to_constant = FxHashMap::default();
        name_to_constant.insert(String::from("b0"), false);

        let tables_1d = vec![Table1D::new(vec![false])];
        let mut name_to_table_1d = FxHashMap::default();
        name_to_table_1d.insert(String::from("b1"), 0);

        let tables_2d = vec![Table2D::new(vec![vec![false]])];
        let mut name_to_table_2d = FxHashMap::default();
        name_to_table_2d.insert(String::from("b2"), 0);

        let tables_3d = vec![Table3D::new(vec![vec![vec![false]]])];
        let mut name_to_table_3d = FxHashMap::default();
        name_to_table_3d.insert(String::from("b3"), 0);

        let tables = vec![Table::new(FxHashMap::default(), false)];
        let mut name_to_table = FxHashMap::default();
        name_to_table.insert(String::from("b4"), 0);

        let bool_tables = dypdl::TableData {
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

        let mut table_names = Array::new();
        let mut table_values = Hash::new();
        let mut dictionary_names = Array::new();
        let mut dictionary_values = Hash::new();
        let result = table_data_to_yaml(
            &bool_tables,
            "boolean",
            &mut table_names,
            &mut table_values,
            &mut dictionary_names,
            &mut dictionary_values,
        );
        assert!(result.is_ok());

        let mut expected_table_names = Array::new();
        let mut b0 = Hash::new();
        b0.insert(Yaml::from_str("name"), Yaml::from_str("b0"));
        b0.insert(Yaml::from_str("type"), Yaml::from_str("boolean"));
        expected_table_names.push(Yaml::Hash(b0));

        let mut b1 = Hash::new();
        b1.insert(Yaml::from_str("name"), Yaml::from_str("b1"));
        b1.insert(Yaml::from_str("type"), Yaml::from_str("boolean"));
        b1.insert(Yaml::from_str("args"), Yaml::Array(vec![Yaml::Integer(1)]));
        expected_table_names.push(Yaml::Hash(b1));

        let mut b2 = Hash::new();
        b2.insert(Yaml::from_str("name"), Yaml::from_str("b2"));
        b2.insert(Yaml::from_str("type"), Yaml::from_str("boolean"));
        b2.insert(
            Yaml::from_str("args"),
            Yaml::Array(vec![Yaml::Integer(1), Yaml::Integer(1)]),
        );
        expected_table_names.push(Yaml::Hash(b2));

        let mut b3 = Hash::new();
        b3.insert(Yaml::from_str("name"), Yaml::from_str("b3"));
        b3.insert(Yaml::from_str("type"), Yaml::from_str("boolean"));
        b3.insert(
            Yaml::from_str("args"),
            Yaml::Array(vec![Yaml::Integer(1), Yaml::Integer(1), Yaml::Integer(1)]),
        );
        expected_table_names.push(Yaml::Hash(b3));

        assert_eq!(table_names, expected_table_names);

        let mut expected_dictionary_names = Array::new();
        let mut b4 = Hash::new();
        b4.insert(Yaml::from_str("name"), Yaml::from_str("b4"));
        b4.insert(Yaml::from_str("type"), Yaml::from_str("boolean"));
        b4.insert(Yaml::from_str("default"), Yaml::Boolean(false));
        expected_dictionary_names.push(Yaml::Hash(b4));

        assert_eq!(dictionary_names, expected_dictionary_names);

        let mut expected_table_values = Hash::new();
        expected_table_values.insert(Yaml::from_str("b0"), Yaml::Boolean(false));
        let mut b1_values = Hash::new();
        b1_values.insert(Yaml::Integer(0), Yaml::Boolean(false));
        expected_table_values.insert(Yaml::from_str("b1"), Yaml::Hash(b1_values));
        let mut b2_values = Hash::new();
        b2_values.insert(
            Yaml::Array(vec![Yaml::Integer(0), Yaml::Integer(0)]),
            Yaml::Boolean(false),
        );
        expected_table_values.insert(Yaml::from_str("b2"), Yaml::Hash(b2_values));
        let mut b3_values = Hash::new();
        b3_values.insert(
            Yaml::Array(vec![Yaml::Integer(0), Yaml::Integer(0), Yaml::Integer(0)]),
            Yaml::Boolean(false),
        );
        expected_table_values.insert(Yaml::from_str("b3"), Yaml::Hash(b3_values));

        assert_eq!(table_values, expected_table_values);

        let mut expected_dictionary_values = Hash::new();
        expected_dictionary_values.insert(Yaml::from_str("b4"), Yaml::Hash(Hash::new()));

        assert_eq!(dictionary_values, expected_dictionary_values);
    }

    #[test]
    fn element_table_data_to_yaml() {
        let mut name_to_constant = FxHashMap::default();
        name_to_constant.insert(String::from("e0"), 0);

        let tables_1d = vec![Table1D::new(vec![0])];
        let mut name_to_table_1d = FxHashMap::default();
        name_to_table_1d.insert(String::from("e1"), 0);

        let tables_2d = vec![Table2D::new(vec![vec![0]])];
        let mut name_to_table_2d = FxHashMap::default();
        name_to_table_2d.insert(String::from("e2"), 0);

        let tables_3d = vec![Table3D::new(vec![vec![vec![0]]])];
        let mut name_to_table_3d = FxHashMap::default();
        name_to_table_3d.insert(String::from("e3"), 0);

        let tables = vec![Table::new(FxHashMap::default(), 0)];
        let mut name_to_table = FxHashMap::default();
        name_to_table.insert(String::from("e4"), 0);

        let element_tables = dypdl::TableData {
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

        let mut table_names = Array::new();
        let mut table_values = Hash::new();
        let mut dictionary_names = Array::new();
        let mut dictionary_values = Hash::new();
        let result = table_data_to_yaml(
            &element_tables,
            "element",
            &mut table_names,
            &mut table_values,
            &mut dictionary_names,
            &mut dictionary_values,
        );
        assert!(result.is_ok());

        let mut expected_table_names = Array::new();
        let mut e0 = Hash::new();
        e0.insert(Yaml::from_str("name"), Yaml::from_str("e0"));
        e0.insert(Yaml::from_str("type"), Yaml::from_str("element"));
        expected_table_names.push(Yaml::Hash(e0));

        let mut e1 = Hash::new();
        e1.insert(Yaml::from_str("name"), Yaml::from_str("e1"));
        e1.insert(Yaml::from_str("type"), Yaml::from_str("element"));
        e1.insert(Yaml::from_str("args"), Yaml::Array(vec![Yaml::Integer(1)]));
        expected_table_names.push(Yaml::Hash(e1));

        let mut e2 = Hash::new();
        e2.insert(Yaml::from_str("name"), Yaml::from_str("e2"));
        e2.insert(Yaml::from_str("type"), Yaml::from_str("element"));
        e2.insert(
            Yaml::from_str("args"),
            Yaml::Array(vec![Yaml::Integer(1), Yaml::Integer(1)]),
        );
        expected_table_names.push(Yaml::Hash(e2));

        let mut e3 = Hash::new();
        e3.insert(Yaml::from_str("name"), Yaml::from_str("e3"));
        e3.insert(Yaml::from_str("type"), Yaml::from_str("element"));
        e3.insert(
            Yaml::from_str("args"),
            Yaml::Array(vec![Yaml::Integer(1), Yaml::Integer(1), Yaml::Integer(1)]),
        );
        expected_table_names.push(Yaml::Hash(e3));

        assert_eq!(table_names, expected_table_names);

        let mut expected_dictionary_names = Array::new();
        let mut e4 = Hash::new();
        e4.insert(Yaml::from_str("name"), Yaml::from_str("e4"));
        e4.insert(Yaml::from_str("type"), Yaml::from_str("element"));
        e4.insert(Yaml::from_str("default"), Yaml::Integer(0));
        expected_dictionary_names.push(Yaml::Hash(e4));

        assert_eq!(dictionary_names, expected_dictionary_names);

        let mut expected_table_values = Hash::new();
        expected_table_values.insert(Yaml::from_str("e0"), Yaml::Integer(0));
        let mut e1_values = Hash::new();
        e1_values.insert(Yaml::Integer(0), Yaml::Integer(0));
        expected_table_values.insert(Yaml::from_str("e1"), Yaml::Hash(e1_values));
        let mut e2_values = Hash::new();
        e2_values.insert(
            Yaml::Array(vec![Yaml::Integer(0), Yaml::Integer(0)]),
            Yaml::Integer(0),
        );
        expected_table_values.insert(Yaml::from_str("e2"), Yaml::Hash(e2_values));
        let mut e3_values = Hash::new();
        e3_values.insert(
            Yaml::Array(vec![Yaml::Integer(0), Yaml::Integer(0), Yaml::Integer(0)]),
            Yaml::Integer(0),
        );
        expected_table_values.insert(Yaml::from_str("e3"), Yaml::Hash(e3_values));

        assert_eq!(table_values, expected_table_values);

        let mut expected_dictionary_values = Hash::new();
        expected_dictionary_values.insert(Yaml::from_str("e4"), Yaml::Hash(Hash::new()));

        assert_eq!(dictionary_values, expected_dictionary_values);
    }

    #[test]
    fn set_table_data_to_yaml_ok() {
        let mut name_to_constant = FxHashMap::default();
        name_to_constant.insert(String::from("s0"), Set::with_capacity(1));

        let tables_1d = vec![Table1D::new(vec![Set::with_capacity(1)])];
        let mut name_to_table_1d = FxHashMap::default();
        name_to_table_1d.insert(String::from("s1"), 0);

        let tables_2d = vec![Table2D::new(vec![vec![Set::with_capacity(1)]])];
        let mut name_to_table_2d = FxHashMap::default();
        name_to_table_2d.insert(String::from("s2"), 0);

        let tables_3d = vec![Table3D::new(vec![vec![vec![Set::with_capacity(1)]]])];
        let mut name_to_table_3d = FxHashMap::default();
        name_to_table_3d.insert(String::from("s3"), 0);

        let tables = vec![Table::new(FxHashMap::default(), Set::with_capacity(1))];
        let mut name_to_table = FxHashMap::default();
        name_to_table.insert(String::from("s4"), 0);

        let set_tables = dypdl::TableData {
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

        let mut table_names = Array::new();
        let mut table_values = Hash::new();
        let mut dictionary_names = Array::new();
        let mut dictionary_values = Hash::new();
        let result = set_table_data_to_yaml(
            &set_tables,
            "set",
            &mut table_names,
            &mut table_values,
            &mut dictionary_names,
            &mut dictionary_values,
        );
        assert!(result.is_ok());

        let mut expected_table_names = Array::new();
        let mut s0 = Hash::new();
        s0.insert(Yaml::from_str("name"), Yaml::from_str("s0"));
        s0.insert(Yaml::from_str("type"), Yaml::from_str("set"));
        s0.insert(Yaml::from_str("object"), Yaml::Integer(1));
        expected_table_names.push(Yaml::Hash(s0));

        let mut s1 = Hash::new();
        s1.insert(Yaml::from_str("name"), Yaml::from_str("s1"));
        s1.insert(Yaml::from_str("type"), Yaml::from_str("set"));
        s1.insert(Yaml::from_str("args"), Yaml::Array(vec![Yaml::Integer(1)]));
        s1.insert(Yaml::from_str("object"), Yaml::Integer(1));
        expected_table_names.push(Yaml::Hash(s1));

        let mut s2 = Hash::new();
        s2.insert(Yaml::from_str("name"), Yaml::from_str("s2"));
        s2.insert(Yaml::from_str("type"), Yaml::from_str("set"));
        s2.insert(
            Yaml::from_str("args"),
            Yaml::Array(vec![Yaml::Integer(1), Yaml::Integer(1)]),
        );
        s2.insert(Yaml::from_str("object"), Yaml::Integer(1));
        expected_table_names.push(Yaml::Hash(s2));

        let mut s3 = Hash::new();
        s3.insert(Yaml::from_str("name"), Yaml::from_str("s3"));
        s3.insert(Yaml::from_str("type"), Yaml::from_str("set"));
        s3.insert(
            Yaml::from_str("args"),
            Yaml::Array(vec![Yaml::Integer(1), Yaml::Integer(1), Yaml::Integer(1)]),
        );
        s3.insert(Yaml::from_str("object"), Yaml::Integer(1));
        expected_table_names.push(Yaml::Hash(s3));

        assert_eq!(table_names, expected_table_names);

        let mut expected_dictionary_names = Array::new();
        let mut s4 = Hash::new();
        s4.insert(Yaml::from_str("name"), Yaml::from_str("s4"));
        s4.insert(Yaml::from_str("type"), Yaml::from_str("set"));
        s4.insert(Yaml::from_str("object"), Yaml::Integer(1));
        s4.insert(Yaml::from_str("default"), Yaml::Array(vec![]));
        expected_dictionary_names.push(Yaml::Hash(s4));

        assert_eq!(dictionary_names, expected_dictionary_names);

        let mut expected_table_values = Hash::new();
        expected_table_values.insert(Yaml::from_str("s0"), Yaml::Array(vec![]));
        let mut s1_values = Hash::new();
        s1_values.insert(Yaml::Integer(0), Yaml::Array(vec![]));
        expected_table_values.insert(Yaml::from_str("s1"), Yaml::Hash(s1_values));
        let mut s2_values = Hash::new();
        s2_values.insert(
            Yaml::Array(vec![Yaml::Integer(0), Yaml::Integer(0)]),
            Yaml::Array(vec![]),
        );
        expected_table_values.insert(Yaml::from_str("s2"), Yaml::Hash(s2_values));
        let mut s3_values = Hash::new();
        s3_values.insert(
            Yaml::Array(vec![Yaml::Integer(0), Yaml::Integer(0), Yaml::Integer(0)]),
            Yaml::Array(vec![]),
        );
        expected_table_values.insert(Yaml::from_str("s3"), Yaml::Hash(s3_values));

        assert_eq!(table_values, expected_table_values);

        let mut expected_dictionary_values = Hash::new();
        expected_dictionary_values.insert(Yaml::from_str("s4"), Yaml::Hash(Hash::new()));

        assert_eq!(dictionary_values, expected_dictionary_values);
    }

    #[test]
    fn set_table_data_to_yaml_err() {
        let mut name_to_constant = FxHashMap::default();
        name_to_constant.insert(String::from("s0"), Set::with_capacity(1));

        // Empty Table1D without element.
        let tables_1d = vec![Table1D::new(vec![])];
        let mut name_to_table_1d = FxHashMap::default();
        name_to_table_1d.insert(String::from("s1"), 0);

        let tables_2d = vec![Table2D::new(vec![vec![Set::with_capacity(1)]])];
        let mut name_to_table_2d = FxHashMap::default();
        name_to_table_2d.insert(String::from("s2"), 0);

        let tables_3d = vec![Table3D::new(vec![vec![vec![Set::with_capacity(1)]]])];
        let mut name_to_table_3d = FxHashMap::default();
        name_to_table_3d.insert(String::from("s3"), 0);

        let tables = vec![Table::new(FxHashMap::default(), Set::with_capacity(1))];
        let mut name_to_table = FxHashMap::default();
        name_to_table.insert(String::from("s4"), 0);

        let set_tables = dypdl::TableData {
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

        let mut table_names = Array::new();
        let mut table_values = Hash::new();
        let mut dictionary_names = Array::new();
        let mut dictionary_values = Hash::new();
        let result = set_table_data_to_yaml(
            &set_tables,
            "set",
            &mut table_names,
            &mut table_values,
            &mut dictionary_names,
            &mut dictionary_values,
        );
        assert!(result.is_err());
    }
}
