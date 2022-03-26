use crate::table;
use approx::{AbsDiffEq, RelativeEq};
use rustc_hash::{FxHashMap, FxHashSet};

#[derive(Debug, PartialEq, Clone, Default)]
pub struct TableData<T> {
    pub name_to_constant: FxHashMap<String, T>,
    pub tables_1d: Vec<table::Table1D<T>>,
    pub name_to_table_1d: FxHashMap<String, usize>,
    pub tables_2d: Vec<table::Table2D<T>>,
    pub name_to_table_2d: FxHashMap<String, usize>,
    pub tables_3d: Vec<table::Table3D<T>>,
    pub name_to_table_3d: FxHashMap<String, usize>,
    pub tables: Vec<table::Table<T>>,
    pub name_to_table: FxHashMap<String, usize>,
}

impl<T> TableData<T> {
    pub fn get_name_set(&self) -> FxHashSet<String> {
        let mut name_set = FxHashSet::default();
        for name in self.name_to_constant.keys() {
            name_set.insert(name.clone());
        }
        for name in self.name_to_table_1d.keys() {
            name_set.insert(name.clone());
        }
        for name in self.name_to_table_2d.keys() {
            name_set.insert(name.clone());
        }
        for name in self.name_to_table_3d.keys() {
            name_set.insert(name.clone());
        }
        for name in self.name_to_table.keys() {
            name_set.insert(name.clone());
        }
        name_set
    }
}

impl<T: AbsDiffEq> AbsDiffEq for TableData<T>
where
    T::Epsilon: Copy,
{
    type Epsilon = T::Epsilon;

    fn default_epsilon() -> T::Epsilon {
        T::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: T::Epsilon) -> bool {
        if self.name_to_constant.len() != other.name_to_constant.len() {
            return false;
        }
        for (key, x) in &self.name_to_constant {
            match other.name_to_constant.get(key) {
                Some(y) if x.abs_diff_eq(y, epsilon) => {}
                _ => return false,
            }
        }
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

impl<T: RelativeEq> RelativeEq for TableData<T>
where
    T::Epsilon: Copy,
{
    fn default_max_relative() -> T::Epsilon {
        T::default_max_relative()
    }

    fn relative_eq(&self, other: &Self, epsilon: T::Epsilon, max_relative: T::Epsilon) -> bool {
        if self.name_to_constant.len() != other.name_to_constant.len() {
            return false;
        }
        for (key, x) in &self.name_to_constant {
            match other.name_to_constant.get(key) {
                Some(y) if x.relative_eq(y, epsilon, max_relative) => {}
                _ => return false,
            }
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::{assert_relative_eq, assert_relative_ne};

    #[test]
    fn get_name_set() {
        let mut name_to_constant = FxHashMap::default();
        name_to_constant.insert(String::from("i0"), 0);

        let tables_1d = vec![table::Table1D::new(vec![10, 20, 30])];
        let mut name_to_table_1d = FxHashMap::default();
        name_to_table_1d.insert(String::from("i1"), 0);

        let tables_2d = vec![table::Table2D::new(vec![
            vec![10, 20, 30],
            vec![10, 10, 10],
            vec![10, 10, 10],
        ])];
        let mut name_to_table_2d = FxHashMap::default();
        name_to_table_2d.insert(String::from("i2"), 0);

        let tables_3d = vec![table::Table3D::new(vec![
            vec![vec![10, 20, 30], vec![0, 0, 0], vec![0, 0, 0]],
            vec![vec![0, 0, 0], vec![0, 0, 0], vec![0, 0, 0]],
            vec![vec![0, 0, 0], vec![0, 0, 0], vec![0, 0, 0]],
        ])];
        let mut name_to_table_3d = FxHashMap::default();
        name_to_table_3d.insert(String::from("i3"), 0);

        let mut map = FxHashMap::default();
        let key = vec![0, 1, 0, 0];
        map.insert(key, 100);
        let key = vec![0, 1, 0, 1];
        map.insert(key, 200);
        let key = vec![0, 1, 2, 0];
        map.insert(key, 300);
        let key = vec![0, 1, 2, 1];
        map.insert(key, 400);
        let tables = vec![table::Table::new(map, 0)];
        let mut name_to_table = FxHashMap::default();
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
        let mut expected = FxHashSet::default();
        expected.insert(String::from("i0"));
        expected.insert(String::from("i1"));
        expected.insert(String::from("i2"));
        expected.insert(String::from("i3"));
        expected.insert(String::from("i4"));
        assert_eq!(integer_tables.get_name_set(), expected);
    }

    #[test]
    fn relative_eq() {
        let mut name_to_table_1d = FxHashMap::default();
        name_to_table_1d.insert(String::from("t0"), 0);
        name_to_table_1d.insert(String::from("t1"), 1);
        let t1 = TableData {
            tables_1d: vec![
                table::Table1D::new(vec![1.0, 2.0]),
                table::Table1D::new(vec![2.0, 3.0]),
            ],
            name_to_table_1d: name_to_table_1d.clone(),
            ..Default::default()
        };
        let t2 = TableData {
            tables_1d: vec![
                table::Table1D::new(vec![1.0, 2.0]),
                table::Table1D::new(vec![2.0, 3.0]),
            ],
            name_to_table_1d: name_to_table_1d.clone(),
            ..Default::default()
        };
        assert_relative_eq!(t1, t2);
        let t2 = TableData {
            tables_1d: vec![
                table::Table1D::new(vec![1.0, 2.0]),
                table::Table1D::new(vec![3.0, 3.0]),
            ],
            name_to_table_1d,
            ..Default::default()
        };
        assert_relative_ne!(t1, t2);
        let mut name_to_table_1d = FxHashMap::default();
        name_to_table_1d.insert(String::from("t0"), 0);
        let t2 = TableData {
            tables_1d: vec![table::Table1D::new(vec![1.0, 2.0])],
            name_to_table_1d,
            ..Default::default()
        };
        assert_relative_ne!(t1, t2);
    }
}
